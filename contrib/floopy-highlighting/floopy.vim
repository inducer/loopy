" Vim highlighting for Floopy (Fortran+Loopy) source code
" -------------------------------------------------------
" Installation:
" Just drop this file into ~/.vim/syntax/floopy.vim
"
" Then do
" :set filetype=floopy
"
" You may also include a line
" vim: filetype=pyopencl.python
" at the end of your file to set the file type automatically.
"
" Another option is to include the following in your .vimrc
" au BufRead,BufNewFile *.floopy set filetype=floopy

runtime! syntax/fortran.vim

unlet b:current_syntax
try
  syntax include @clCode syntax/opencl.vim
catch
  syntax include @clCode syntax/c.vim
endtry

if exists('b:current_syntax')
  let s:current_syntax=b:current_syntax
  " Remove current syntax definition, as some syntax files (e.g. cpp.vim)
  " do nothing if b:current_syntax is defined.
  unlet b:current_syntax
endif

syntax include @LoopyPython syntax/python.vim
try
  syntax include @LoopyPython after/syntax/python.vim
catch
endtry

if exists('s:current_syntax')
  let b:current_syntax=s:current_syntax
else
  unlet b:current_syntax
endif

syntax region textSnipLoopyPython
\ matchgroup=Comment
\ start='$loopy begin transform' end='$loopy end transform'
\ containedin=ALL
\ contains=@LoopyPython
