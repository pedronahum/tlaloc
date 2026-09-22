package io.tlaloc.core

/**
 * Layer 2 §0.4.243+ — typed buffer handle for step boundaries.
 *
 * A [BufferHandle] is the only thing that crosses a [io.tlaloc.maestro.MaestroStep]
 * boundary; never a materialized tensor, never an untyped reference. The
 * type parameters carry compile-time-known shape ([T] = the [DTensor] type
 * with named axes from Layer 1) and mesh placement ([M] = a phantom
 * [Mesh]).
 *
 * # Lifecycle (v1 — single-threaded)
 *
 * Reference counting via [HandleRef]. The producing step bumps to 1; each
 * step that consumes the handle increments before reading and decrements
 * after; `close()` (or explicit [release]) decrements. When the count
 * reaches zero the underlying runtime buffer is released.
 *
 * v1 assumes **sequential workflow execution** — refcounting is a plain
 * `var`, not lock-free. The workflow builder generates explicit increment
 * / release calls in one logical thread. Multi-threaded refcount safety
 * is a v2 enhancement, deferred until Layer 3 introduces parallel dispatch.
 *
 * # Why a value class
 *
 * `@JvmInline value class BufferHandle<T, M>(val ref: HandleRef)` erases at
 * the JVM level to a bare `HandleRef` — no allocation overhead at the
 * call site, and the phantom type parameters [T] and [M] cost nothing at
 * runtime. They exist purely to flow through Kotlin's type checker so
 * step composition and reshard insertion are compile-time decisions.
 */
@ExperimentalTlalocApi
@JvmInline
value class BufferHandle<T : DTensor<*, *>, M : Mesh>(val ref: HandleRef) : AutoCloseable {

    /** Increment the refcount. Call before exposing to a downstream consumer. */
    fun retain(): BufferHandle<T, M> {
        ref.retain()
        return this
    }

    /** Decrement the refcount. When it hits zero the underlying buffer is released. */
    fun release() {
        ref.release()
    }

    /** [AutoCloseable] adapter so `use { … }` works on a handle. */
    override fun close() {
        release()
    }

    /** Whether the handle is still live (refcount > 0). */
    val isLive: Boolean get() = ref.refCount > 0
}

/**
 * Backing store for [BufferHandle]'s reference count. Carries an opaque
 * runtime-buffer identifier ([nativeId]), a payload slot that v1 uses as
 * a stand-in for a real runtime buffer pool, and the refcount.
 *
 * # Payload (v1 stub)
 *
 * Layer 2 v1 carries the materialized [DTensor] (or whatever the producing
 * step actually computes) directly in [payload]. This is a deliberate v1
 * shortcut — Layer 3 introduces a real runtime buffer pool (PJRT / IREE
 * device buffers) that the stub executor and the runtime image can both
 * point at. Tracked as an open question in the audit.
 *
 * # Refcount semantics
 *
 * - Constructed at refcount = 1 (the producing step holds the initial reference).
 * - [retain] increments before exposing to a new consumer.
 * - [release] decrements; on transition to 0, the [onZero] callback fires
 *   (this is where Layer 3's dispatcher will free the underlying buffer).
 *
 * **Not thread-safe.** v1 contract: workflow execution is sequential.
 */
@ExperimentalTlalocApi
class HandleRef(
    val nativeId: Long,
    val payload: Any? = null,
    private val onZero: () -> Unit = {},
) {
    private var count: Int = 1

    /** Current reference count. Strictly nonnegative. */
    val refCount: Int get() = count

    fun retain() {
        check(count > 0) { "cannot retain a released HandleRef (nativeId=$nativeId)" }
        count++
    }

    fun release() {
        check(count > 0) { "cannot release an already-released HandleRef (nativeId=$nativeId)" }
        count--
        if (count == 0) onZero()
    }

    override fun toString(): String = "HandleRef(nativeId=$nativeId, refCount=$count)"
}
