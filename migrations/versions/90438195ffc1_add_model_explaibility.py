"""add model explainability

Revision ID: 90438195ffc1
Revises: fba24b0cf0f8
Create Date: 2025-12-28 14:53:54.378708
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = "90438195ffc1"
down_revision: Union[str, Sequence[str], None] = "fba24b0cf0f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "modelexplainability",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("global_importance", sa.JSON(), nullable=False),
        sa.Column("local_explanation", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["modelartifact.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("model_id"),
    )

    op.create_index(
        "ix_modelexplainability_model_id",
        "modelexplainability",
        ["model_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_modelexplainability_model_id",
        table_name="modelexplainability",
    )
    op.drop_table("modelexplainability")
