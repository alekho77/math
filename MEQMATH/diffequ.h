//---------------------------------------------------------------------------
#ifndef diffequH
#define diffequH
//---------------------------------------------------------------------------
/* Указатель на функцию вида y'=f(x,y) */
typedef Extended __fastcall (*HDiffEqu)(Extended&, Extended&);
/* Решение задачи Коши методом Эйлера */
Extended __fastcall SolEiler(const HDiffEqu f,const Extended& Xo,const Extended& Yo,const Extended& X,const Cardinal& m,Extended& Eps);
//---------------------------------------------------------------------------
#endif
