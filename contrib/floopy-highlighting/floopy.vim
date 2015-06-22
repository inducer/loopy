" Vim highlighting for Floopy (Fortran+Loopy) source code
" -------------------------------------------------------
" Installation:
" Just drop this file into ~/.vim/syntax/floopy.vim
"
" Then do
" :set filetype=floopy
"
" You may also include a line
" vim: filetype=floopy.python
" at the end of your file to set the file type automatically.
"
" Another option is to include the following in your .vimrc
" au BufRead,BufNewFile *.floopy set filetype=floopy

runtime! syntax/fortran.vim

unlet b:current_syntax
syntax include @LoopyPython syntax/python.vim

if exists('s:current_syntax')
  let b:current_syntax=s:current_syntax
else
  unlet b:current_syntax
endif

syntax region textSnipLoopyPython
\ matchgroup=Comment
\ start='$loopy begin' end='$loopy end'
\ containedin=ALL
\ contains=@LoopyPython
