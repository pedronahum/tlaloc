/**
 * THIS FILE IS SUPPOSED TO FAIL TO COMPILE.
 *
 * It is not commented out and it is not a string: it is real Kotlin in a real
 * source set, and the build has a task whose whole job is to try to compile it
 * and show you what the compiler says:
 *
 *     ./gradlew -p examples/quickstart shapeError
 *
 * What is wrong with it: `contract` sums over the axis the two operands share
 * by NAME. Here the first operand's axes are named (Batch, SeqLen) and the
 * second's are (Hidden, Hidden) — they share no axis name, so there is nothing
 * to contract over. In NumPy, PyTorch or JAX this is a runtime error at best
 * and a silently wrong answer at worst. Here it is a build failure, and in the
 * IDE it is a red squiggle under the call while you type.
 */
import io.tlaloc.autograd.grad2
import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Hidden
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.ops.contract
import io.tlaloc.core.ops.sum

fun shapeErrorThatMustNotCompile() {
    grad2 { a: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>,
            b: DTensor<Rank2<Named<Hidden, Sym>, Named<Hidden, Sym>>, F32> ->
        (a contract b).sum()
    }
}
