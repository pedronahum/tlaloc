/**
 * THIS FILE IS SUPPOSED TO FAIL TO COMPILE.
 *
 * It is real Kotlin in a real source set — not a comment, not a string — and
 * the build has a task whose only job is to compile it and show you the
 * rejection:
 *
 *     ./gradlew -p examples/named-indices shapeError
 *
 * `a`'s axis 1 is named `SeqLen`. `b`'s axis 0 is named `Vocab`. No overload of
 * `contract` matches operands that share no axis name, so KOTLIN'S OWN TYPE
 * CHECKER rejects the call — no Tlaloc compiler plugin is involved here at all.
 * This project does not even put the plugin on its classpath.
 *
 * That is the design goal rather than a limitation: the contraction rule is
 * expressed in ordinary Kotlin generics, so the IDE draws the red squiggle with
 * no Tlaloc-specific tooling installed.
 */
import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Hidden
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.Vocab
import io.tlaloc.core.ops.contract

fun shapeErrorThatMustNotCompile(
    a: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>,
    b: DTensor<Rank2<Named<Vocab, Sym>, Named<Hidden, Sym>>, F32>,
) {
    // Contract a (Batch x SeqLen) with b (Vocab x Hidden): SeqLen != Vocab.
    val c = a contract b
    println(c.dims.toList())
}
