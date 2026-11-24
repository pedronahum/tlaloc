package brooklyn

import kotlin.test.Test
import kotlin.test.assertEquals

class MainTest {
    @Test
    fun everyStageIsReachable() {
        val names = listOf(
            "models", "snapshot", "ingest", "tools", "tasks", "agent", "baseline", "teach", "probe",
            "train", "distill", "rft", "eval", "serve", "chat",
        )
        assertEquals(names, STAGES.map { it.name })
        // Only the stages not built yet are dispatched: a built stage does real work.
        for (stage in STAGES.filter { it.run == null }) assertEquals(NOT_BUILT, dispatch(listOf(stage.name)), stage.name)
    }

    @Test
    fun unknownSubcommandFails() = assertEquals(2, dispatch(listOf("nope")))

    @Test
    fun sqliteWithFts5Loads() = assertEquals(0, check())
}
