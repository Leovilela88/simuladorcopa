"""Estrutura da Copa do Mundo 2026.

48 seleções, 12 grupos (A..L) de 4 times → 1º, 2º e 8 melhores 3ºs avançam
para os 32-avos (R32). Daí mata-mata até a final. 104 jogos no total.
"""
from __future__ import annotations

from dataclasses import dataclass

GROUPS = list("ABCDEFGHIJKL")  # 12 grupos


@dataclass(frozen=True)
class Slot:
    """Vaga em jogo de mata-mata, referencia placeholder até ser resolvida.

    Ex.: '1A' = 1º colocado do grupo A; '3ACD' = 3º colocado de A, C ou D
    (a regra exata de cruzamento dos 8 melhores 3ºs depende de quais grupos
    avançam, ver FIFA). 'W49' = vencedor do jogo 49.
    """
    code: str


# Cronograma da fase de grupos: 12 grupos × 3 rodadas × 2 jogos = 72 jogos.
def group_stage_fixtures() -> list[tuple[str, int, tuple[int, int]]]:
    """Retorna lista de (group, matchday, (idx_home, idx_away)).

    Convenção de rodadas (índices 1..4 dentro do grupo):
      MD1: 1v2, 3v4
      MD2: 1v3, 4v2
      MD3: 4v1, 2v3
    """
    schedule = [
        (1, (1, 2)), (1, (3, 4)),
        (2, (1, 3)), (2, (4, 2)),
        (3, (4, 1)), (3, (2, 3)),
    ]
    out = []
    for g in GROUPS:
        for md, pair in schedule:
            out.append((g, md, pair))
    return out


# Esqueleto dos 32-avos (R32). A FIFA confirmou o chaveamento em 2023:
# https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026
# Aqui listamos confrontos como pares de placeholders 1X/2X/3XYZ.
# OBS: o casamento dos 8 melhores 3ºs depende da combinação que avança —
# isso é resolvido em runtime pela função `resolve_third_places`.
R32_BRACKET: list[tuple[str, str]] = [
    ("1A", "3CDEFGHIJKLAB?1"),
    ("2C", "3DEFGHIJKL?2"),
    ("1B", "2A"),
    ("1F", "2C?"),
    # Esqueleto simplificado — refinaremos com o chaveamento oficial.
    # Mantemos 16 confrontos. Para o MVP, podemos gerar dinamicamente.
]


def empty_groups() -> dict[str, list[str | None]]:
    """Retorna {grupo: [None, None, None, None]} pronto para popular."""
    return {g: [None] * 4 for g in GROUPS}
