# Copyright 2026 Vinay Williams

"""Control package.

Importing this package runs the ``@register`` decorators on every concrete
control module so the registry is fully populated before
:func:`painting_assist.controls.registry.create_all` is called.

Adding a new control is a two-step change:

1. Create ``controls/<your_control>.py`` with a ``@register`` decorated
   :class:`~painting_assist.controls.base.Control` subclass.
2. Add one import line below.

No other file needs to change.
"""

from __future__ import annotations

from . import crop  # noqa: F401  -- registers CropControl
from . import white_balance  # noqa: F401  -- registers WhiteBalanceControl
from . import adjust  # noqa: F401  -- registers ToneControl
from . import blur  # noqa: F401  -- registers BlurControl
from . import quantize  # noqa: F401  -- registers ColourGroupsControl
from . import values  # noqa: F401  -- registers ValuesControl
from . import temperature  # noqa: F401  -- registers TemperatureMapControl
from . import grid  # noqa: F401  -- registers GridControl
from . import flip  # noqa: F401  -- registers FlipControl
# add one import line here per new control
