"""Persist optional book illustration settings, generations, scopes, attempts, and registry (SQL-01)."""

from alembic import op
import sqlalchemy as sa


revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "illustration_settings",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("book_id", sa.String(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("settings_sha256", sa.String(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_illustration_settings"),
        sa.ForeignKeyConstraint(
            ["book_id"],
            ["books.id"],
            name="fk_illustration_settings_book_id_books",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("book_id", "revision", name="uq_illustration_settings_book_id_revision"),
        sa.CheckConstraint("revision > 0", name="ck_illustration_settings_revision_positive"),
    )

    op.create_table(
        "illustration_generations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("book_id", sa.String(), nullable=False),
        sa.Column("generation_number", sa.Integer(), nullable=False),
        sa.Column("settings_id", sa.String(), nullable=False),
        sa.Column("request_key", sa.String(), nullable=False),
        sa.Column("source_sha256", sa.String(), nullable=False),
        sa.Column("source_authority_json", sa.JSON(), nullable=False),
        sa.Column("requested_sections_json", sa.JSON(), nullable=False),
        sa.Column("manifest_json", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("manifest_sha256", sa.String(), nullable=True),
        sa.Column("registry_sha256", sa.String(), nullable=True),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("reason_code", sa.String(), nullable=True),
        sa.Column("calls_reserved", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("calls_sent", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_illustration_generations"),
        sa.ForeignKeyConstraint(
            ["book_id"],
            ["books.id"],
            name="fk_illustration_generations_book_id_books",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["settings_id"],
            ["illustration_settings.id"],
            name="fk_illustration_generations_settings_id_illustration_settings",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "book_id",
            "generation_number",
            name="uq_illustration_generations_book_id_generation_number",
        ),
        sa.CheckConstraint(
            "generation_number > 0",
            name="ck_illustration_generations_generation_number_positive",
        ),
        sa.CheckConstraint(
            "state IN ('waiting_source','queued','running','ready','failed','skipped','revoked')",
            name="ck_illustration_generations_state_valid",
        ),
        sa.CheckConstraint(
            "calls_reserved >= 0",
            name="ck_illustration_generations_calls_reserved_nonnegative",
        ),
        sa.CheckConstraint(
            "calls_sent >= 0",
            name="ck_illustration_generations_calls_sent_nonnegative",
        ),
    )
    op.create_index(
        "uq_illustration_generations_active_request",
        "illustration_generations",
        ["book_id", "request_key"],
        unique=True,
        sqlite_where=sa.text("state != 'revoked'"),
    )
    op.create_index(
        "ix_illustration_generation_history",
        "illustration_generations",
        ["book_id", "generation_number", "created_at"],
    )
    op.create_index(
        "ix_illustration_generation_settings",
        "illustration_generations",
        ["settings_id"],
    )
    op.create_index(
        "ix_illustration_generation_registry",
        "illustration_generations",
        ["book_id", "registry_sha256"],
    )

    op.create_table(
        "illustration_heads",
        sa.Column("book_id", sa.String(), nullable=False),
        sa.Column("settings_id", sa.String(), nullable=True),
        sa.Column("active_generation_id", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("book_id", name="pk_illustration_heads"),
        sa.ForeignKeyConstraint(
            ["book_id"],
            ["books.id"],
            name="fk_illustration_heads_book_id_books",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["settings_id"],
            ["illustration_settings.id"],
            name="fk_illustration_heads_settings_id_illustration_settings",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["active_generation_id"],
            ["illustration_generations.id"],
            name="fk_illustration_heads_active_generation_id_illustration_generations",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_illustration_heads_settings", "illustration_heads", ["settings_id"])
    op.create_index(
        "ix_illustration_heads_active_generation",
        "illustration_heads",
        ["active_generation_id"],
    )

    op.create_table(
        "illustration_scopes",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("generation_id", sa.String(), nullable=False),
        sa.Column("parent_scope_id", sa.String(), nullable=True),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("scope_key", sa.String(), nullable=False),
        sa.Column("section_id", sa.String(), nullable=True),
        sa.Column("section_order", sa.Integer(), nullable=True),
        sa.Column("section_title", sa.String(), nullable=True),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("input_sha256", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("reason_code", sa.String(), nullable=True),
        sa.Column("attempt_id", sa.String(), nullable=True),
        sa.Column("output_sha256", sa.String(), nullable=True),
        sa.Column("payload_json", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("summary_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_illustration_scopes"),
        sa.ForeignKeyConstraint(
            ["generation_id"],
            ["illustration_generations.id"],
            name="fk_illustration_scopes_generation_id_illustration_generations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_scope_id"],
            ["illustration_scopes.id"],
            name="fk_illustration_scopes_parent_scope_id_illustration_scopes",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "generation_id",
            "stage",
            "scope_key",
            name="uq_illustration_scopes_generation_id_stage_scope_key",
        ),
        sa.CheckConstraint(
            "stage IN ('extract','alias','design','fact_scope','candidates','selection','prompt','chapter')",
            name="ck_illustration_scopes_stage_valid",
        ),
        sa.CheckConstraint(
            "state IN ('pending','running','accepted','failed','skipped','split')",
            name="ck_illustration_scopes_state_valid",
        ),
    )
    op.create_index(
        "ix_illustration_scopes_generation",
        "illustration_scopes",
        ["generation_id", "stage", "state", "section_id"],
    )
    op.create_index("ix_illustration_scopes_parent", "illustration_scopes", ["parent_scope_id"])
    op.create_index(
        "ix_illustration_scopes_reuse",
        "illustration_scopes",
        ["stage", "input_sha256", "state"],
    )
    op.create_index("ix_illustration_scopes_task", "illustration_scopes", ["task_id"])

    op.create_table(
        "illustration_attempts",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("request_sha256", sa.String(), nullable=True),
        sa.Column("request_meta_json", sa.JSON(), nullable=False),
        sa.Column("outcome_json", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_illustration_attempts"),
        sa.ForeignKeyConstraint(
            ["scope_id"],
            ["illustration_scopes.id"],
            name="fk_illustration_attempts_scope_id_illustration_scopes",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("scope_id", "ordinal", name="uq_illustration_attempts_scope_id_ordinal"),
        sa.CheckConstraint("ordinal > 0", name="ck_illustration_attempts_ordinal_positive"),
        sa.CheckConstraint(
            "state IN ('reserved','sent','accepted','rejected','failed','interrupted','stale','cancelled')",
            name="ck_illustration_attempts_state_valid",
        ),
    )
    op.create_index(
        "uq_illustration_attempts_active_scope",
        "illustration_attempts",
        ["scope_id"],
        unique=True,
        sqlite_where=sa.text("state IN ('reserved', 'sent')"),
    )
    op.create_index(
        "ix_illustration_attempt_scope",
        "illustration_attempts",
        ["scope_id", "state"],
    )
    op.create_index("ix_illustration_attempt_state", "illustration_attempts", ["state"])
    op.create_index("ix_illustration_attempt_task", "illustration_attempts", ["task_id"])

    op.create_table(
        "illustration_registry_items",
        sa.Column("generation_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("item_key", sa.String(), nullable=False),
        sa.Column("entity_id", sa.String(), nullable=False),
        sa.Column("position_key", sa.String(), nullable=False),
        sa.Column("is_back_applicable", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("payload_sha256", sa.String(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("generation_id", "kind", "item_key", name="pk_illustration_registry_items"),
        sa.ForeignKeyConstraint(
            ["generation_id"],
            ["illustration_generations.id"],
            name="fk_illustration_registry_items_generation_id_illustration_generations",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "kind IN ('entity','fact','alias_decision','design')",
            name="ck_illustration_registry_items_kind_valid",
        ),
        sa.CheckConstraint(
            "is_back_applicable IN (0, 1)",
            name="ck_illustration_registry_items_is_back_applicable_bool",
        ),
    )
    op.create_index(
        "ix_illustration_registry_entity",
        "illustration_registry_items",
        ["generation_id", "entity_id", "kind", "position_key"],
    )
    op.create_index(
        "ix_illustration_registry_back",
        "illustration_registry_items",
        ["generation_id", "entity_id", "is_back_applicable"],
    )


def downgrade() -> None:
    op.drop_table("illustration_registry_items")
    op.drop_table("illustration_attempts")
    op.drop_table("illustration_scopes")
    op.drop_table("illustration_heads")
    op.drop_table("illustration_generations")
    op.drop_table("illustration_settings")
