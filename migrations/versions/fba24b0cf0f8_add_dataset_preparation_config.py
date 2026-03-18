"""add dataset preparation config

Revision ID: fba24b0cf0f8
Revises: 983bbb2a7a62
Create Date: 2025-12-28 11:00:13.896866
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "fba24b0cf0f8"
down_revision: Union[str, Sequence[str], None] = "983bbb2a7a62"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -------------------------------------------------
    # DATASET PREPARATION CONFIG TABLE
    # -------------------------------------------------
    op.create_table(
        "dataset_preparation_config",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),

        sa.Column("problem_type", sa.String(), nullable=False),
        sa.Column("target", sa.String(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),

        sa.Column("test_size", sa.Float(), nullable=False),
        sa.Column("stratify", sa.Boolean(), nullable=False),

        sa.Column("encoding", sa.String(), nullable=True),
        sa.Column("scaling", sa.String(), nullable=True),

        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),

        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset.id"],
            ondelete="CASCADE",
        ),
    )

    op.create_index(
        "ix_dataset_preparation_config_dataset_id",
        "dataset_preparation_config",
        ["dataset_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_dataset_preparation_config_dataset_id",
        table_name="dataset_preparation_config",
    )
    op.drop_table("dataset_preparation_config")
