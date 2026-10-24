"""Translation guidance note for glossary decisions (PRD § 11).

A translation term alone is insufficient when a book-level decision affects text agreement:
character grammatical gender in the target language, title declension, or register. The note
resides alongside the decision—on the analysis candidate and the approved concept—and is sent
to the model together with the translation variant.

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
    """Apply the migration."""
    with op.batch_alter_table('term_concepts', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('note', sa.Text(), nullable=False, server_default='')
        )
    with op.batch_alter_table('glossary_candidates', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('note', sa.Text(), nullable=False, server_default='')
        )


def downgrade() -> None:
    """Roll back the migration."""
    with op.batch_alter_table('glossary_candidates', schema=None) as batch_op:
        batch_op.drop_column('note')
    with op.batch_alter_table('term_concepts', schema=None) as batch_op:
        batch_op.drop_column('note')
