"""Переводческое указание при решении глоссария (PRD § 11).

Одного варианта перевода мало там, где решение уровня книги влияет на согласование
в тексте: род персонажа в целевом языке, склонение названия, регистр речи. Указание
живёт рядом с решением — у кандидата анализа и у утверждённого понятия — и уходит
модели вместе с вариантом перевода.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Применяет миграцию."""
    with op.batch_alter_table('term_concepts', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('note', sa.Text(), nullable=False, server_default='')
        )
    with op.batch_alter_table('glossary_candidates', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('note', sa.Text(), nullable=False, server_default='')
        )


def downgrade() -> None:
    """Откатывает миграцию."""
    with op.batch_alter_table('glossary_candidates', schema=None) as batch_op:
        batch_op.drop_column('note')
    with op.batch_alter_table('term_concepts', schema=None) as batch_op:
        batch_op.drop_column('note')
