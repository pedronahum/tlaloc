package brooklyn

import io.tlaloc.core.io.jsonQuote
import java.sql.DriverManager
import kotlin.system.exitProcess

/**
 * Brooklyn Eats: one entry point, one subcommand per pipeline stage.
 *
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
    Stage("ingest", "snapshot the Brooklyn sources into build/data/brooklyn.db", ::ingest),
    Stage("tools", "tool server over the snapshot (snapshot | live | record | replay)"),
    Stage("tasks", "generate questions with ground truth; split by restaurant"),
    Stage("agent", "run the agent loop against tlaloc-endpoint"),
    Stage("baseline", "base TwIL-8B and Ripple-65B on dev and test"),
    Stage("teach", "teacher trajectories from Ripple-65B on the training split"),
    Stage("probe", "LoRA memory and step-time probe per sequence bucket"),
    Stage("train", "LoRA SFT of TwIL-8B"),
    Stage("distill", "distillation from Ripple's top-k log-probabilities"),
    Stage("rft", "rejection-sampling round"),
    Stage("eval", "evaluation on the test split"),
    Stage("serve", "merge, export and serve the tuned student"),
    Stage("chat", "chat client with live tools"),
)

/** Exit code of a stage that is not implemented yet. */
const val NOT_BUILT = 3

fun usage(): String = buildString {
    appendLine("usage: brooklyn-eats <subcommand> [args]")
    appendLine()
    for (s in STAGES) appendLine("  %-9s %s".format(s.name, s.summary))
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
    exitProcess(dispatch(args.toList()))
}
