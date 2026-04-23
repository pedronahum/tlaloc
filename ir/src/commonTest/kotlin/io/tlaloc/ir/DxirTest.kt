package io.tlaloc.ir

import io.tlaloc.core.F32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DxirTest {
    private val f32v = DxirType(F32, listOf(4))
    private val f32s = DxirType(F32, emptyList())

    @Test
    fun typeFormatting() {
        assertEquals("f32[4]", f32v.toString())
        assertEquals("f32", f32s.toString())
        assertEquals("f32[2,3,4]", DxirType(F32, listOf(2, 3, 4)).toString())
    }

    @Test
    fun typeElementCount() {
        assertEquals(1L, f32s.elementCount)
        assertEquals(4L, f32v.elementCount)
        assertEquals(24L, DxirType(F32, listOf(2, 3, 4)).elementCount)
    }

    @Test
    fun builderAssignsDistinctIds() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32v)
            val y = op(OpKind.RELU, listOf(x), f32v)
            val z = op(OpKind.SUM, listOf(y), f32s)
            listOf(z)
        }
        val ids = (fn.params + fn.body).map { it.id }
        assertEquals(ids.size, ids.toSet().size)
    }

    @Test
    fun functionValidatesReferencedIds() {
        val orphan = DxirParam(id = 99, name = "ghost", type = f32v)
        val p = DxirParam(id = 0, name = "x", type = f32v)
        assertFailsWith<IllegalArgumentException> {
            DxirFunction(
                name = "broken",
                params = listOf(p),
                body = listOf(DxirOp(1, OpKind.RELU, listOf(orphan), emptyMap(), f32v)),
                returns = emptyList(),
            )
        }
    }

    @Test
    fun moduleLookupByName() {
        val f = DxirBuilder.function("identity") {
            val x = param("x", f32v)
            listOf(x)
        }
        val g = DxirBuilder.function("negate") {
            val x = param("x", f32v)
            listOf(op(OpKind.NEG, listOf(x), f32v))
        }
        val m = DxirModule(listOf(f, g))
        assertEquals("identity", m.function("identity")?.name)
        assertEquals("negate", m.function("negate")?.name)
        assertEquals(null, m.function("missing"))
    }

    @Test
    fun prettyPrintsFunction() {
        val fn = DxirBuilder.function("small") {
            val x = param("x", f32v)
            val y = op(OpKind.RELU, listOf(x), f32v)
            val s = op(OpKind.SUM, listOf(y), f32s)
            listOf(s)
        }
        val out = DxirModule(listOf(fn)).pretty()
        assertTrue(out.contains("fn small"), "module output: $out")
        assertTrue(out.contains("relu(%0)"))
        assertTrue(out.contains("sum(%1)"))
        assertTrue(out.contains("return %2"))
    }

    @Test
    fun callNodeReferencesCallee() {
        val helper = DxirBuilder.function("helper") {
            val x = param("x", f32v)
            listOf(op(OpKind.RELU, listOf(x), f32v))
        }
        val main = DxirBuilder.function("main") {
            val x = param("x", f32v)
            listOf(call(helper, listOf(x), f32v))
        }
        val callNode = main.body.filterIsInstance<DxirCall>().single()
        assertEquals("helper", callNode.callee.name)
    }

    @Test
    fun multiResultOpExposesEachResultViaResultIndex() {
        val fn = DxirBuilder.function("splitting") {
            val x = param("x", DxirType(F32, listOf(6, 4)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(DxirType(F32, listOf(3, 4)), DxirType(F32, listOf(3, 4))),
                attrs = mapOf("axis" to 0, "sizes" to listOf(3, 3)),
            )
            listOf(split.result(0), split.result(1))
        }
        val split = fn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.SPLIT }
        assertEquals(2, split.numResults)
        assertTrue(split.isMultiResult)
        assertEquals(2, fn.returns.size)
        val r0 = fn.returns[0]
        val r1 = fn.returns[1]
        assertTrue(r0 is DxirOpResult && r0.index == 0)
        assertTrue(r1 is DxirOpResult && r1.index == 1)
        assertEquals(split.id, r0.id)
        assertEquals(split.id, r1.id)
    }

    @Test
    fun singleResultOpResultZeroReturnsSelf() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32v)
            val y = op(OpKind.RELU, listOf(x), f32v)
            listOf(y.result(0))
        }
        val returned = fn.returns.single()
        val relu = fn.body.filterIsInstance<DxirOp>().single()
        // For single-result ops, result(0) === the op itself (no wrapper).
        assertTrue(returned === relu)
    }

    @Test
    fun resultIndexOutOfBoundsIsRejected() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32v)
            val y = op(OpKind.RELU, listOf(x), f32v)
            listOf(y)
        }
        val relu = fn.body.filterIsInstance<DxirOp>().single()
        assertFailsWith<IllegalArgumentException> { relu.result(1) }
    }

    @Test
    fun regionBuilderCreatesSingleBlockWithArgsAndBody() {
        val region = DxirBuilder.function("outer") {
            val x = param("x", f32v)
            val mc = op(
                OpKind.MANUAL_COMPUTATION, listOf(x), f32v,
                regions = listOf(
                    region {
                        val local = arg(f32v)
                        val y = op(OpKind.RELU, listOf(local), f32v)
                        yields(y)
                    },
                ),
            )
            listOf(mc)
        }.body.filterIsInstance<DxirOp>().single { it.op == OpKind.MANUAL_COMPUTATION }
            .regions.single()
        assertEquals(1, region.blocks.size)
        val block = region.entryBlock
        assertEquals(1, block.args.size)
        assertEquals(1, block.body.size)
        assertEquals(1, block.terminator.size)
    }

    @Test
    fun nestedRegionIdsShareOuterBuilderCounter() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32v)
            val mc = op(
                OpKind.MANUAL_COMPUTATION, listOf(x), f32v,
                regions = listOf(
                    region {
                        val a = arg(f32v)
                        val y = op(OpKind.NEG, listOf(a), f32v)
                        yields(y)
                    },
                ),
            )
            listOf(mc)
        }
        // All ids across outer + nested should be unique.
        val outerIds = fn.body.filterIsInstance<DxirOp>().map { it.id }
        val innerIds = fn.body.filterIsInstance<DxirOp>()
            .flatMap { it.regions }
            .flatMap { it.blocks }
            .flatMap { b -> b.args.map { it.id } + b.body.map { it.id } }
        val allIds = (fn.params.map { it.id } + outerIds + innerIds)
        assertEquals(allIds.size, allIds.toSet().size, "nested ids must all be unique")
    }

    @Test
    fun nestedBodyReferencesValidateCorrectly() {
        // Inner op references a block arg (declared inside) — validation should accept.
        DxirBuilder.function("ok") {
            val x = param("x", f32v)
            val mc = op(
                OpKind.MANUAL_COMPUTATION, listOf(x), f32v,
                regions = listOf(
                    region {
                        val a = arg(f32v)
                        val r = op(OpKind.RELU, listOf(a), f32v)
                        yields(r)
                    },
                ),
            )
            listOf(mc)
        }
        // No exception = pass.
    }

    @Test
    fun printerShowsNestedRegionWithBlock() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32v)
            val mc = op(
                OpKind.MANUAL_COMPUTATION, listOf(x), f32v,
                regions = listOf(
                    region {
                        val a = arg(f32v)
                        val y = op(OpKind.NEG, listOf(a), f32v)
                        yields(y)
                    },
                ),
            )
            listOf(mc)
        }
        val out = DxirModule(listOf(fn)).pretty()
        assertTrue(out.contains("block ("), "block header missing: $out")
        assertTrue(out.contains("yield"), "yield terminator missing: $out")
    }

    @Test
    fun multiResultPrinterShowsPackedForm() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(10)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(DxirType(F32, listOf(6)), DxirType(F32, listOf(4))),
                attrs = mapOf("axis" to 0, "sizes" to listOf(6, 4)),
            )
            listOf(split.result(0), split.result(1))
        }
        val out = DxirModule(listOf(fn)).pretty()
        assertTrue(out.contains(":2"), "packed-form `%N:2` missing: $out")
        assertTrue(out.contains("#0"), "result ref `%N#0` missing: $out")
        assertTrue(out.contains("#1"), "result ref `%N#1` missing: $out")
    }

    // ---- B.0a: structural validation for IF / WHILE ops -----------------

    private val boolS = DxirType(io.tlaloc.core.Bool, emptyList())

    @Test
    fun ifBuilderConstructsTwoRegionShape() {
        val fn = DxirBuilder.function("ifFn") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifop = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val y = op(OpKind.NEG, listOf(x), f32s)
                    yields(y)
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifop)
        }
        val ifop = fn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.IF }
        assertEquals(1, ifop.operands.size)
        assertEquals(2, ifop.regions.size)
        assertEquals(0, ifop.regions[0].entryBlock.args.size)
        assertEquals(0, ifop.regions[1].entryBlock.args.size)
    }

    @Test
    fun ifWithNonBoolPredicateRejected() {
        assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("badIf") {
                val x = param("x", f32s)
                val ifop = ifOp(
                    cond = x, // f32 — not Bool
                    types = listOf(f32s),
                    thenRegion = region { yields(x) },
                    elseRegion = region { yields(x) },
                )
                listOf(ifop)
            }
        }
    }

    @Test
    fun ifWithMismatchedBranchTypesRejected() {
        assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("badIf") {
                val x = param("x", f32s)
                val pred = op(OpKind.STEP, listOf(x), boolS)
                val wrongTypeConst = const(0.0, DxirType(io.tlaloc.core.F64, emptyList()))
                val ifop = ifOp(
                    cond = pred,
                    types = listOf(f32s),
                    thenRegion = region { yields(x) },
                    elseRegion = region { yields(wrongTypeConst) }, // f64 ≠ declared f32
                )
                listOf(ifop)
            }
        }
    }

    @Test
    fun whileBuilderConstructsTwoRegionShapeWithBlockArgs() {
        val fn = DxirBuilder.function("loopy") {
            val xInit = param("x", f32s)
            val w = whileOp(
                inits = listOf(xInit),
                cond = { args ->
                    // Trip count predicate: x > 0 (always true here; just a shape test).
                    val p = op(OpKind.STEP, listOf(args[0]), boolS)
                    yields(p)
                },
                body = { args ->
                    val nx = op(OpKind.NEG, listOf(args[0]), f32s)
                    yields(nx)
                },
            )
            listOf(w)
        }
        val w = fn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.WHILE }
        assertEquals(1, w.operands.size)
        assertEquals(1, w.types.size)
        assertEquals(2, w.regions.size)
        assertEquals(1, w.regions[0].entryBlock.args.size)  // condArgs
        assertEquals(1, w.regions[1].entryBlock.args.size)  // bodyArgs
        assertEquals(1, w.regions[0].entryBlock.terminator.size)  // predicate
        assertEquals(1, w.regions[1].entryBlock.terminator.size)  // back-edge
    }

    @Test
    fun whileWithNonBoolPredicateRejected() {
        assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("badWhile") {
                val xInit = param("x", f32s)
                val w = whileOp(
                    inits = listOf(xInit),
                    cond = { args -> yields(args[0]) }, // f32 — not Bool
                    body = { args -> yields(args[0]) },
                )
                listOf(w)
            }
        }
    }

    @Test
    fun whileWithMismatchedBodyTerminatorTypeRejected() {
        assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("badWhile") {
                val xInit = param("x", f32s)
                val wrongTypeConst = const(0.0, DxirType(io.tlaloc.core.F64, emptyList()))
                val w = whileOp(
                    inits = listOf(xInit),
                    cond = { args ->
                        val p = op(OpKind.STEP, listOf(args[0]), boolS)
                        yields(p)
                    },
                    body = { _ -> yields(wrongTypeConst) }, // f64 ≠ declared f32
                )
                listOf(w)
            }
        }
    }
}
