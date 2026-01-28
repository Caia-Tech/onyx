from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 style reversible byte<->unicode mapping.

    This maps each byte 0..255 to a unique unicode character so that byte-level
    BPE can operate on unicode strings without losing information.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def unicode_to_bytes() -> dict[str, int]:
    enc = bytes_to_unicode()
    return {v: k for k, v in enc.items()}


@dataclass(frozen=True)
class Byte256Tokenizer:
    """
    Minimal 256-token byte tokenizer (ids 0..255).

    - `encode(text)` returns UTF-8 bytes as integer ids.
    - `decode(ids)` turns ids back into UTF-8 text.
    """

    encoding: str = "utf-8"
    errors: str = "strict"

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        return list(text.encode(self.encoding, errors=self.errors))

    def decode(self, ids: Iterable[int], *, errors: Optional[str] = None) -> str:
        err = self.errors if errors is None else errors
        b = bytes(int(i) & 0xFF for i in ids)
        return b.decode(self.encoding, errors=err)


def _smoke() -> None:
    enc = bytes_to_unicode()
    dec = unicode_to_bytes()
    assert len(enc) == 256
    assert len(dec) == 256
    for b in range(256):
        ch = enc[b]
        assert dec[ch] == b

    t = Byte256Tokenizer(errors="strict")
    s = "hello naïve café\n"
    assert t.decode(t.encode(s)) == s


if __name__ == "__main__":
    _smoke()
