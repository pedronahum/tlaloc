package brooklyn

import kotlin.test.Test
import kotlin.test.assertEquals

class MainTest {
    @Test
    fun everyStageIsReachable() {
        val names = listOf(
            "ingest", "tools", "tasks", "agent", "baseline", "teach", "probe",
            "train", "distill", "rft", "eval", "serve", "chat",
        )
        assertEquals(names, STAGES.map { it.name })
        for (n in names) {
            val code = dispatch(listOf(n))
            val stage = STAGES.first { it.name == n }
            if (stage.run == null) assertEquals(NOT_BUILT, code, n)
        }
    }

    @Test
    fun unknownSubcommandFails() = assertEquals(2, dispatch(listOf("nope")))

    @Test
    fun sqliteWithFts5Loads() = assertEquals(0, check())
}
