Reference: Variable Name Registry
=================================

Reserved Kernel Variable Names
------------------------------

The variable name prefix ``_lp`` is reserved for internal usage; when creating
kernel variables, users should *not* use names beginning with ``_lp``. This
prefix is used for certain variables created when operating on Loopy's kernel
IR. For Loopy developers, further information on name prefixes used within
submodules is below.

Variable Name Registry
----------------------

Some Loopy submodules append a sub-prefix to the ``_lp`` prefix for
internally-created variable names. These prefixes should only be used for names
created within the listed submodule.

Reserved Name Prefixes
^^^^^^^^^^^^^^^^^^^^^^

=============== =============================
Prefix          Module
=============== =============================
``_lp``         :mod:`loopy`
``_lp_sched``   loopy.linearization.checker
=============== =============================
