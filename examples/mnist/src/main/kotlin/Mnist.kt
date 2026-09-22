/**
 * The dataset: the real MNIST, fetched once and cached.
 *
 * Nothing here is Tlaloc-specific — it is 70 lines of "read the idx format" so
 * that the rest of the example can be about the gradient rather than about
 * plumbing. The files land in `~/.cache/tlaloc-datasets/mnist/` and are reused
 * by every later run.
 *
 * If there is no network and no cache, [load] returns null and the example
 * prints a named SKIP and exits 0 — the house rule for examples: degrade
 * honestly, never stack-trace on a laptop.
 */
import java.io.DataInputStream
import java.io.File
import java.net.URI
import java.util.zip.GZIPInputStream

/** Images as row-major [n, 784] floats in [0,1]; labels as ints in 0..9. */
class MnistSplit(val n: Int, val images: FloatArray, val labels: IntArray) {
    /** One-hot targets, row-major [n, 10] — the training signal. */
    fun oneHot(): FloatArray = FloatArray(n * 10).also { out ->
        for (i in 0 until n) out[i * 10 + labels[i]] = 1f
    }

    /** A contiguous slice of [count] rows starting at [from], wrapping around. */
    fun batch(from: Int, count: Int): Pair<FloatArray, FloatArray> {
        val x = FloatArray(count * 784)
        val y = FloatArray(count * 10)
        for (k in 0 until count) {
            val i = (from + k) % n
            System.arraycopy(images, i * 784, x, k * 784, 784)
            y[k * 10 + labels[i]] = 1f
        }
        return x to y
    }
}

class Mnist(val train: MnistSplit, val test: MnistSplit)

object MnistLoader {
    private const val MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    private val CACHE = File(System.getProperty("user.home"), ".cache/tlaloc-datasets/mnist")

    private val FILES = listOf(
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )

    /** Where the cache lives, for the SKIP message. */
    val cacheDir: File get() = CACHE

    /** The curl one-liner a reader can run by hand if this machine has no network. */
    fun manualFetch(): String =
        "mkdir -p ${CACHE.path} && cd ${CACHE.path} && " +
            FILES.joinToString(" && ") { "curl -sSLO $MIRROR$it" }

    /**
     * Load MNIST, downloading anything missing. Returns null (with a printed
     * reason) when the data is neither cached nor reachable.
     */
    fun load(): Mnist? {
        CACHE.mkdirs()
        for (f in FILES) {
            val target = File(CACHE, f)
            if (target.isFile && target.length() > 0) continue
            try {
                print("    fetching $f ... ")
                URI.create(MIRROR + f).toURL().openStream().use { inp ->
                    target.outputStream().use { out -> inp.copyTo(out) }
                }
                println("%,d bytes".format(target.length()))
            } catch (t: Throwable) {
                println("FAILED (${t::class.simpleName}: ${t.message})")
                target.delete()
                return null
            }
        }
        return Mnist(
            train = split(File(CACHE, FILES[0]), File(CACHE, FILES[1])),
            test = split(File(CACHE, FILES[2]), File(CACHE, FILES[3])),
        )
    }

    private fun split(imageGz: File, labelGz: File): MnistSplit {
        val (n, images) = readImages(imageGz)
        val labels = readLabels(labelGz)
        require(labels.size == n) { "label/image count mismatch: ${labels.size} vs $n" }
        return MnistSplit(n, images, labels)
    }

    /** idx3: magic 0x00000803, then count, rows, cols, then bytes. */
    private fun readImages(gz: File): Pair<Int, FloatArray> =
        DataInputStream(GZIPInputStream(gz.inputStream().buffered())).use { d ->
            require(d.readInt() == 0x00000803) { "${gz.name}: not an idx3 image file" }
            val n = d.readInt()
            val rows = d.readInt()
            val cols = d.readInt()
            require(rows == 28 && cols == 28) { "${gz.name}: expected 28x28, got ${rows}x$cols" }
            val raw = ByteArray(n * rows * cols)
            d.readFully(raw)
            // Unsigned byte -> [0,1]. That scaling is the only preprocessing.
            n to FloatArray(raw.size) { (raw[it].toInt() and 0xFF) / 255f }
        }

    /** idx1: magic 0x00000801, then count, then one byte per label. */
    private fun readLabels(gz: File): IntArray =
        DataInputStream(GZIPInputStream(gz.inputStream().buffered())).use { d ->
            require(d.readInt() == 0x00000801) { "${gz.name}: not an idx1 label file" }
            val n = d.readInt()
            val raw = ByteArray(n)
            d.readFully(raw)
            IntArray(n) { raw[it].toInt() and 0xFF }
        }
}
