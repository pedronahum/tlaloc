package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Proxy
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

/**
 * PhiCalculus falls back to the unsimplified function when the symbolic
 * engine throws an exception, and lets a JVM [Error] (out of memory, a
 * linkage failure) propagate instead of hiding it behind the fallback.
 */
class PhiCalculusErrorPropagationTest {

    private val f32 = DxirType(F32, emptyList())

    private fun mulByOne() = DxirBuilder.function("mul_by_one_err") {
        val x = param("x", f32)
        listOf(op(OpKind.MUL, listOf(const(1L, f32), x), f32))
    }

    private class InjectedError : Error("injected")

    /** A real Symja engine whose `simplify` throws [failure]. */
    private fun engineThrowing(failure: Throwable): SymbolicEngine {
        val real = SymjaEngine()
        return Proxy.newProxyInstance(
            SymbolicEngine::class.java.classLoader,
            arrayOf(SymbolicEngine::class.java),
        ) { _, method, args ->
            if (method.name == "simplify") throw failure
            try {
                method.invoke(real, *(args ?: emptyArray()))
            } catch (e: InvocationTargetException) {
                throw e.targetException
            }
        } as SymbolicEngine
    }

    @Test
    fun anEngineExceptionFallsBackToTheOriginalFunction() {
        val fn = mulByOne()
        assertSame(fn, PhiCalculus.simplifyReturns(fn, engineThrowing(IllegalStateException("engine failed"))))
    }

    @Test
    fun anEngineErrorPropagates() {
        assertFailsWith<InjectedError> { PhiCalculus.simplifyReturns(mulByOne(), engineThrowing(InjectedError())) }
    }
}
