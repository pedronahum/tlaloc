#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""An OpenAI-compatible endpoint in front of a decoupled speculative Tlaloc model
served by Triton, so that clients and benchmark harnesses written for vLLM run
against it unchanged.

  POST /v1/chat/completions   messages, the model's chat template (chat_template_kwargs)
  POST /v1/completions        a prompt string
  GET  /v1/models
  GET  /stats                 requests, prefix reuse, tokens per step

Both POST routes take max_tokens (or max_completion_tokens), stream,
stream_options.include_usage, ignore_eos and stop_token_ids. Decoding is greedy:
the backend verifies drafts greedily, so a request with temperature > 0 is
refused (--allow-sampling serves it greedily instead).

The backend keeps a sequence's KV between requests. An OpenAI client resends the
whole conversation, so each live sequence remembers the tokens its KV holds (the
prompt and the generated tokens but the last, which stays pending); a request
whose tokens start with them sends only the rest. A chat request whose messages
extend a live conversation's (its last reply included) sends the template's text
for the new messages after the reply as it was generated: a chat template need
not render an earlier turn as it rendered the last (Qwen's drops the empty think
block), so the tokens would not match. Otherwise the request starts a new
sequence, ending the least recently used idle one when the model holds no more.

  openai_server.py --triton localhost:8031 --model qwen36 --tokenizer Qwen/Qwen3.6-35B-A3B --port 8100
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequence_client import SequenceClient


class Live:
    """A backend sequence, the tokens its KV holds and, for a chat, its conversation."""

    def __init__(self, corrid):
        self.corrid = corrid
        self.kv = []
        self.pending = None  # the last generated token, not yet run
        self.chat = None  # (messages with the reply, template kwargs, the end token's text or None)
        self.busy = True
        self.used = time.monotonic()


class Sequences:
    """The live sequences, matched to requests by token prefix."""

    def __init__(self, url, model, limit, state_file):
        self.url, self.model, self.limit, self.state_file = url, model, limit, state_file
        self.cond = threading.Condition()
        self.live = []
        self.ids = itertools.count(int(time.time()) % 100000 * 100 + 1)
        # The backend keeps a sequence until Triton ends it, so the sequences an
        # earlier endpoint left open would hold the model's slots: end them.
        try:
            for corrid in json.load(open(state_file)):
                self.end(Live(corrid))
        except (OSError, ValueError):
            pass
        self.save()

    def save(self):
        """The live sequences' ids, for the next endpoint to end (called under the lock or alone)."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump([s.corrid for s in self.live], f)
        except OSError as e:
            print(f"cannot write {self.state_file}: {e}", file=sys.stderr)

    def acquire(self, ids, extend=None):
        """(live, tokens to send, start): the idle sequence with the longest KV
        prefix of `ids`, else one that `extend(live)` continues (the tokens to
        send, or None), else a new one. Triton holds at most `limit` sequences and
        queues a new one until another ends, so a new sequence waits for an idle
        one to end (the least recently used) when `limit` are live."""
        with self.cond:
            while True:
                best = None
                for s in self.live:
                    n = len(s.kv)
                    if not s.busy and 0 < n < len(ids) and ids[:n] == s.kv and (best is None or n > len(best.kv)):
                        best = s
                if best is not None:
                    best.busy = True
                    return best, ids[len(best.kv):], False
                if extend is not None:
                    for s in sorted((s for s in self.live if not s.busy and s.chat), key=lambda s: -len(s.kv)):
                        send = extend(s)
                        if send is not None:
                            s.busy = True
                            return s, send, False
                idle = sorted((s for s in self.live if not s.busy), key=lambda s: s.used)
                if len(self.live) < self.limit or idle:
                    break
                self.cond.wait()
            evict = None
            if len(self.live) >= self.limit:
                evict = idle[0]
                self.live.remove(evict)
            s = Live(next(self.ids))
            self.live.append(s)
            self.save()
        if evict is not None:
            self.end(evict)
        return s, ids, True

    def release(self, s, kv, pending, chat=None):
        with self.cond:
            s.kv, s.pending, s.chat, s.busy, s.used = kv, pending, chat, False, time.monotonic()
            self.cond.notify_all()

    def drop(self, s):
        with self.cond:
            if s in self.live:
                self.live.remove(s)
            self.save()
            self.cond.notify_all()
        self.end(s)

    def end(self, s):
        try:
            c = SequenceClient(self.url, self.model, "grpc")
            c.end_streamed(s.corrid)
            c.close()
        except Exception as e:
            print(f"ending sequence {s.corrid}: {str(e)[:200]}", file=sys.stderr)


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.v = dict(requests=0, reused_requests=0, prompt_tokens=0, reused_tokens=0,
                      completion_tokens=0, steps=0)

    def add(self, **kw):
        with self.lock:
            for k, x in kw.items():
                self.v[k] += x

    def snapshot(self):
        with self.lock:
            v = dict(self.v)
        v["tokens_per_step"] = v["completion_tokens"] / v["steps"] if v["steps"] else None
        return v


class Detokenizer:
    """Text deltas for a growing list of ids; holds back an incomplete UTF-8 character."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], ""

    def add(self, ids):
        self.ids += ids
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[len(self.sent):], text
        return delta


def make_handler(a, tok, seqs, stats, end_ids):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if a.verbose:
                super().log_message(fmt, *args)

        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def error(self, code, msg):
            self.reply(code, {"error": {"message": msg, "type": "invalid_request_error", "code": code}})

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [
                    {"id": a.served_name, "object": "model", "owned_by": "tlaloc", "max_model_len": a.max_model_len}]})
            elif self.path == "/stats":
                self.reply(200, stats.snapshot())
            elif self.path in ("/health", "/v1/health"):
                self.reply(200, {"status": "ok"})
            else:
                self.error(404, f"no route {self.path}")

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self.error(400, "the body is not JSON")
            if self.path == "/v1/chat/completions":
                kw = dict(body.get("chat_template_kwargs") or {})
                try:
                    ids = tok.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=True,
                                                  return_dict=False, **kw)
                except Exception as e:
                    return self.error(400, f"chat template: {e}")
                if isinstance(ids, dict):
                    ids = ids["input_ids"]
                self.complete(body, list(ids), chat=True, kw=kw)
            elif self.path == "/v1/completions":
                p = body.get("prompt")
                if isinstance(p, list) and p and isinstance(p[0], int):
                    ids = p
                elif isinstance(p, str):
                    ids = tok(p, add_special_tokens=False)["input_ids"]
                else:
                    return self.error(400, "prompt is one string or one list of token ids")
                self.complete(body, list(ids), chat=False)
            else:
                self.error(404, f"no route {self.path}")

        def complete(self, body, ids, chat, kw=None):
            if (body.get("temperature") or 0) > 0 and not a.allow_sampling:
                return self.error(400, "this server decodes greedily; send temperature 0")
            if (body.get("n") or 1) != 1:
                return self.error(400, "n must be 1")
            max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or a.default_max_tokens
            if len(ids) + max_tokens > a.max_model_len:
                return self.error(400, f"{len(ids)} prompt tokens and max_tokens {max_tokens} pass "
                                       f"the model's {a.max_model_len}")
            stops = [] if body.get("ignore_eos") else end_ids + list(body.get("stop_token_ids") or [])
            stream = bool(body.get("stream"))
            usage_in_stream = bool((body.get("stream_options") or {}).get("include_usage"))
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
            created = int(time.time())

            msgs = [(m.get("role"), m.get("content")) for m in body["messages"]] if chat else None

            def extend(s):
                """The tokens that append this chat's new messages to live sequence s, or None."""
                old, old_kw, end_text = s.chat
                n = len(old)
                if old_kw != kw or len(msgs) <= n or msgs[:n] != old or not isinstance(old[-1][1], str):
                    return None
                reply = old[-1][1].strip()
                text = tok.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=False, **kw)
                head = tok.apply_chat_template(body["messages"][:n - 1], add_generation_prompt=True,
                                               tokenize=False, **kw) if n > 1 else ""
                k = text.find(reply, len(head) if text.startswith(head) else 0) if reply else -1
                if k < 0:
                    return None
                after = text[k + len(reply):]
                if end_text is not None:
                    if not after.startswith(end_text):
                        return None
                    after = after[len(end_text):]
                return [s.pending] + tok(after, add_special_tokens=False)["input_ids"]

            s, send, start = seqs.acquire(ids, extend if chat else None)
            reused = 0 if start else len(s.kv)
            ids = list(send) if start else s.kv + list(send)
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            def chunk(delta_text, finish=None, usage=None):
                if chat:
                    delta = {"role": "assistant"} if delta_text is None and finish is None else (
                        {} if delta_text is None else {"content": delta_text})
                    ch = {"index": 0, "delta": delta, "finish_reason": finish}
                    obj = {"id": rid, "object": "chat.completion.chunk", "created": created,
                           "model": a.served_name, "choices": [] if usage else [ch]}
                else:
                    ch = {"index": 0, "text": delta_text or "", "finish_reason": finish}
                    obj = {"id": rid, "object": "text_completion", "created": created,
                           "model": a.served_name, "choices": [] if usage else [ch]}
                if usage:
                    obj["usage"] = usage
                data = f"data: {json.dumps(obj)}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            gen, steps, det, text = [], 0, Detokenizer(tok), []
            c = SequenceClient(a.triton, a.model, "grpc")
            try:
                if stream and chat:
                    chunk(None)  # the role chunk vLLM sends first; carries no content
                for _, step in c.stream_tokens(s.corrid, send, max_tokens, stops, start=start):
                    steps += 1
                    keep = step[:max(0, max_tokens - len(gen))]
                    for i, t in enumerate(keep):
                        if t in stops:
                            keep = keep[:i + 1]
                            break
                    gen += keep
                    shown = [t for t in keep if t not in stops]
                    delta = det.add(shown) if shown else ""
                    text.append(delta)
                    if stream and delta:
                        chunk(delta)
            except Exception as e:
                c.close()
                seqs.drop(s)
                if stream:
                    return self.finish_stream(f"backend: {str(e)[:300]}")
                return self.error(500, f"backend: {str(e)[:300]}")
            c.close()
            finish = "stop" if gen and gen[-1] in stops else "length"
            reply = "".join(text)
            chat_state = None
            if chat and gen:
                end_text = tok.decode([gen[-1]]) if finish == "stop" else None
                chat_state = (msgs + [("assistant", reply)], kw, end_text)
            seqs.release(s, ids + gen[:-1], gen[-1] if gen else None, chat_state)
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(gen),
                     "total_tokens": len(ids) + len(gen),
                     "prompt_tokens_details": {"cached_tokens": reused}}
            stats.add(requests=1, reused_requests=int(reused > 0), prompt_tokens=len(ids),
                      reused_tokens=reused, completion_tokens=len(gen), steps=steps)
            if a.verbose:
                print(f"{rid[:16]} seq {s.corrid} prompt {len(ids)} reused {reused} sent {len(send)} "
                      f"out {len(gen)} steps {steps}", file=sys.stderr)
            if stream:
                chunk(None, finish)
                if usage_in_stream:
                    chunk(None, usage=usage)
                return self.finish_stream()
            out = reply
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": finish}
                obj = {"id": rid, "object": "chat.completion", "created": created, "model": a.served_name,
                       "choices": [choice], "usage": usage}
            else:
                obj = {"id": rid, "object": "text_completion", "created": created, "model": a.served_name,
                       "choices": [{"index": 0, "text": out, "finish_reason": finish}], "usage": usage}
            self.reply(200, obj)

        def finish_stream(self, err=None):
            try:
                if err:
                    data = f"data: {json.dumps({'error': {'message': err}})}\n\n".encode()
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                data = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass  # the client went away

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triton", default="localhost:8001", help="Triton's gRPC address")
    ap.add_argument("--model", required=True, help="the decoupled speculative model in Triton")
    ap.add_argument("--tokenizer", required=True, help="a Hugging Face id or directory with the chat template")
    ap.add_argument("--served-name", default=None, help="the OpenAI model id (default --model)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-sequences", type=int, default=0,
                    help="live sequences kept for prefix reuse (default: the model's max_candidate_sequences)")
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--default-max-tokens", type=int, default=1024)
    ap.add_argument("--state-file", default=None,
                    help="where the live sequences' ids are kept (default ~/.cache/tlaloc/openai-<model>-<port>.json)")
    ap.add_argument("--allow-sampling", action="store_true", help="serve temperature > 0 greedily")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    a.served_name = a.served_name or a.model

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    end_ids = sorted({t for t in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                                  tok.convert_tokens_to_ids("<|endoftext|>")]
                      if isinstance(t, int) and t >= 0 and t != tok.unk_token_id})
    if a.max_sequences <= 0:
        import tritonclient.grpc as tc
        cfg = tc.InferenceServerClient(url=a.triton).get_model_config(a.model, as_json=True)["config"]
        sb = cfg.get("sequence_batching", {})
        a.max_sequences = int(sb.get("oldest", {}).get("max_candidate_sequences") or cfg.get("max_batch_size") or 1)
    state = a.state_file or os.path.expanduser(f"~/.cache/tlaloc/openai-{a.model}-{a.port}.json")
    seqs, stats = Sequences(a.triton, a.model, a.max_sequences, state), Stats()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(a, tok, seqs, stats, end_ids))
    srv.daemon_threads = True
    print(f"OpenAI endpoint on http://{a.host}:{a.port}/v1 for Triton model {a.model} at {a.triton}; "
          f"end tokens {end_ids}, {a.max_sequences} live sequences", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
