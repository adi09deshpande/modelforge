"""dataset storage refactor

Revision ID: 983bbb2a7a62
Revises: 91f4eb3e7387
Create Date: 2025-12-23 11:41:26.775247
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "983bbb2a7a62"
down_revision: Union[str, Sequence[str], None] = "91f4eb3e7387"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ----------------------------
    # DATASET VERSION
    # ----------------------------
    op.add_column(
        "datasetversion",
        sa.Column("file_path", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "datasetversion",
        sa.Column("checksum", sa.String(length=64), nullable=True),
    )

    # remove old binary column
    op.drop_column("datasetversion", "data")

    # ----------------------------
    # MODEL ARTIFACT
    # ----------------------------
    op.add_column(
        "modelartifact",
        sa.Column("file_path", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "modelartifact",
        sa.Column("checksum", sa.String(length=64), nullable=True),
    )

    # remove old columns
    op.drop_column("modelartifact", "filename")
    op.drop_column("modelartifact", "data")


def downgrade() -> None:
    # ----------------------------
    # DATASET VERSION
    # ----------------------------
    op.add_column(
        "datasetversion",
        sa.Column("data", sa.LargeBinary(), nullable=True),
    )
    op.drop_column("datasetversion", "checksum")
    op.drop_column("datasetversion", "file_path")

    # ----------------------------
    # MODEL ARTIFACT
    # ----------------------------
    op.add_column(
        "modelartifact",
        sa.Column("filename", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "modelartifact",
        sa.Column("data", sa.LargeBinary(), nullable=True),
    )
    op.drop_column("modelartifact", "checksum")
    op.drop_column("modelartifact", "file_path")
