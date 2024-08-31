//---------------------------------------------------------------------------
#ifndef diffequH
#define diffequH
//---------------------------------------------------------------------------
/* Pointer to a function of the form y'=f(x,y) */
typedef Extended __fastcall (*HDiffEqu)(Extended&, Extended&);
/* Solving the Cauchy problem using Euler's method */
Extended __fastcall SolEiler(const HDiffEqu f, const Extended& Xo, const Extended& Yo, const Extended& X,
                             const Cardinal& m, Extended& Eps);
//---------------------------------------------------------------------------
#endif
