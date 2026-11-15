package brooklyn

import io.tlaloc.core.io.jsonQuote
import io.tlaloc.endpoint.main as endpointMain
import java.sql.DriverManager
import kotlin.system.exitProcess

/**
 * Brooklyn Eats: one entry point, one subcommand per pipeline stage.
 *
 *   models    the pinned Hugging Face models: download what is missing, check every file's SHA-256
 *   ingest    snapshot OSM, DOHMH inspections, 311 and Google Local reviews into SQLite
 *   tools     the tool server (snapshot / live / record / replay)
 *   tasks     questions with ground truth and their verifiers; train / dev / test splits
 *   agent     the agent loop against tlaloc-endpoint with tools
 *   baseline  base TwIL-8B and Ripple-65B on the dev and test splits
 *   teach     Ripple-65B trajectories on the training split, verified and re-rendered for TwIL
 *   probe     one LoRA step per sequence bucket: peak memory and step time
 *   train     LoRA SFT of TwIL-8B on assistant tokens
 *   distill   distillation from Ripple's top-k log-probabilities
 *   rft       rejection-sampling round
 *   eval      test split, offline tools
 *   serve     merge, export, Triton + tlaloc-endpoint
 *   chat      chat client with live tools
 */
class Stage(val name: String, val summary: String, val run: ((List<String>) -> Int)? = null)

val STAGES: List<Stage> = listOf(
    Stage("models", "download the pinned models into the Hugging Face cache and check their SHA-256", ::models),
    Stage("ingest", "snapshot the Brooklyn sources into build/data/brooklyn.db", ::ingest),
    Stage("tools", "tool server: smoke, schemas, call, serve (snapshot | live | record | replay)", ::tools),
    Stage("tasks", "generate questions with ground truth and their verifier; split by restaurant", ::tasksCmd),
    Stage("agent", "run the agent loop against tlaloc-endpoint (run | parity | show)", ::agent),
    Stage("baseline", "base TwIL-8B and Ripple-65B on dev and test (report | pending; run.sh serves each model)", ::baselineCmd),
    Stage("teach", "teacher trajectories from Ripple-65B on the training split (run | pending | render | check | report)", ::teachCmd),
    Stage("probe", "LoRA memory and step-time probe per sequence bucket"),
    Stage("train", "LoRA SFT of TwIL-8B"),
    Stage("distill", "distillation from Ripple's top-k log-probabilities"),
    Stage("rft", "rejection-sampling round"),
    Stage("eval", "evaluate a served model on a split (run | metrics | spread); tuned models: not built yet", ::evalCmd),
    Stage("serve", "merge, export and serve the tuned student"),
    Stage("chat", "chat client with live tools"),
)

/** Exit code of a stage that is not implemented yet. */
const val NOT_BUILT = 3

fun usage(): String = buildString {
    appendLine("usage: brooklyn-eats <subcommand> [args]")
    appendLine()
    for (s in STAGES) appendLine("  %-9s %s".format(s.name, s.summary))
    appendLine("  %-9s %s".format("verify", "compare inputs and derived files with manifest.json (--update rewrites it)"))
    appendLine("  %-9s %s".format("endpoint", "tlaloc-endpoint's OpenAI API in front of Triton (its own arguments; serving/server.sh starts it)"))
    appendLine("  %-9s %s".format("check", "verify the runtime dependencies load (SQLite, JSON)"))
}

/** Loads the SQLite driver and opens an in-memory database with FTS5; prints the version. */
fun check(): Int {
    DriverManager.getConnection("jdbc:sqlite::memory:").use { c ->
        c.createStatement().use { st ->
            st.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            val v = st.executeQuery("SELECT sqlite_version()").use { rs -> rs.next(); rs.getString(1) }
            println("sqlite ${v}, fts5 ok, json ${jsonQuote("ok")}")
        }
    }
    return 0
}

fun dispatch(args: List<String>): Int {
    val cmd = args.firstOrNull()
    if (cmd == null || cmd == "help" || cmd == "-h" || cmd == "--help") {
        print(usage())
        return if (cmd == null) 2 else 0
    }
    if (cmd == "check") return check()
    if (cmd == "verify") return verify(args.drop(1))
    val stage = STAGES.firstOrNull { it.name == cmd }
    if (stage == null) {
        System.err.println("unknown subcommand: $cmd")
        System.err.print(usage())
        return 2
    }
    val run = stage.run
    if (run == null) {
        System.err.println("${stage.name}: not built yet")
        return NOT_BUILT
    }
    return run(args.drop(1))
}

fun main(args: Array<String>) {
    if (args.firstOrNull() == "endpoint") {
        // The endpoint of this classpath's tlaloc-endpoint, so that the server renders requests with
        // the same template code the trainer and the parity test use. It serves until stopped.
        endpointMain(args.drop(1).toTypedArray())
        Thread.currentThread().join()
        return
    }
    exitProcess(dispatch(args.toList()))
}
