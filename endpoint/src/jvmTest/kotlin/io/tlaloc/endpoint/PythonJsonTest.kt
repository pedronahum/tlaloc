package io.tlaloc.endpoint

import io.tlaloc.core.io.parseJson
import kotlin.test.Test
import kotlin.test.assertEquals

/** Expected strings from CPython 3.12: `json.dumps(json.loads(s), ensure_ascii=False)`. */
class PythonJsonTest {
    private fun dumps(s: String) = pythonJson(parseJson(s))

    @Test fun numbersAsPythonPrintsThem() {
        assertEquals(
            """{"a": 1, "b": [1.5, 2.0, -0.0, 100000.0, 1e-07, 0.1, 1.2345678901234568e+17, 1.25e+20, 3.14159]}""",
            dumps("""{"a": 1, "b": [1.5, 2.0, -0.0, 1e5, 1E-7, 0.1, 123456789012345678.0, 1.25e+20, 3.14159]}"""),
        )
        assertEquals(
            """[1e+16, 1000000000000000.0, 0.0001, 1e-05, 2.5e-05, 100.0, 100.0]""",
            dumps("""[1e16, 1e15, 0.0001, 0.00001, 2.5e-5, 100.0, 1.0e+2]"""),
        )
    }

    @Test fun stringsKeysAndLiterals() {
        assertEquals(
            """{"z": "ü é \n \t \"q\" \\ \u0001", "n": null, "t": true, "big": 12345678901234567890, "neg": 0}""",
            dumps("""{"z": "ü é \n \t \"q\" \\ \u0001", "n": null, "t": true, "big": 12345678901234567890, "neg": -0}"""),
        )
    }
}
