from IPython.core.magic import (magics_class, Magics, cell_magic)

import loopy as lp


@magics_class
class LoopyMagics(Magics):
    @cell_magic
    def fortran_kernel(self, line, cell):
        result = lp.parse_fortran(cell)
        self.shell.user_ns["prog"] = result

    @cell_magic
    def transformed_fortran_kernel(self, line, cell):
        result = lp.parse_transformed_fortran(
                cell,
                transform_code_context=self.shell.user_ns)

        self.shell.user_ns["prog"] = result


def load_ipython_extension(ip):
    ip.register_magics(LoopyMagics)
