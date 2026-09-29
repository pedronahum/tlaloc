package io.tlaloc.plugin

/**
 * What the FIR checker and the IR phase both read off a `vmap` / `vmap2` call: which
 * lambda parameters are batched. `vmap` batches its one argument; `vmap2` takes one
 * marker object per argument (`io.tlaloc.autograd.Batched` or `Broadcast`), given here
 * as the FQNs of the call's argument types (the batch axis first, the lambda last).
 */
internal object VmapCall {
    const val BATCHED = "io.tlaloc.autograd.Batched"
    const val BROADCAST = "io.tlaloc.autograd.Broadcast"

    /** The batched flag per lambda parameter, or null when a marker is missing or none is batched. */
    fun inAxes(callableName: String, argumentTypeFqns: List<String?>): List<Boolean>? {
        if (callableName == "vmap") return listOf(true)
        val flags = (1..2).map { i ->
            when (argumentTypeFqns.getOrNull(i)) {
                BATCHED -> true
                BROADCAST -> false
                else -> return null
            }
        }
        return flags.takeIf { it.any { b -> b } }
    }
}
