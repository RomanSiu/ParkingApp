"""Init

Revision ID: 64f0ad3be463
Revises: 6dba3417da66
Create Date: 2024-07-24 10:10:48.945933

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '64f0ad3be463'
down_revision: Union[str, None] = '6dba3417da66'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('users', 'rate_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('users', 'rate_id',
               existing_type=sa.INTEGER(),
               nullable=True)
    # ### end Alembic commands ###
