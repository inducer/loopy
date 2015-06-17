from __future__ import division

from IPython.core.magic import (magics_class, Magics, cell_magic)

import loopy as lp


@magics_class
class LoopyMagics(Magics):
    @cell_magic
    def fortran_kernel(self, line, cell):
        result = lp.parse_fortran(cell.encode())

        for knl in result:
            self.shell.user_ns[knl.name] = knl

    @cell_magic
    def transformed_fortran_kernel(self, line, cell):
        result = lp.parse_transformed_fortran(
                cell.encode(),
                transform_code_context=self.shell.user_ns)

        for knl in result:
            self.shell.user_ns[knl.name] = knl


def load_ipython_extension(ip):
    ip.register_magics(LoopyMagics)
