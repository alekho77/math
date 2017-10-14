//---------------------------------------------------------------------------
#ifndef integralH
#define integralH
//---------------------------------------------------------------------------
/* Метод интегрирования:
    miRectangle - метод прямоугольников
    miTrapezoid - метод трапеций
    miSimpson   - метод Симпсона
*/
enum TIntegralMethod{miRectangle,miTrapezoid,miSimpson};
//---------------------------------------------------------------------------
/* Тип для обозначения подинтегральной функции */
typedef Extended __fastcall (*HUnderIntFunc)(Extended);
//---------------------------------------------------------------------------
/* Структура для передачи параметров в интегрирующую функцию */
struct TIntPar
{
    HUnderIntFunc F; // Подинтегральная функция
    Extended A,B; // Пределы интегрирования
    Extended Abs; // Абсолютная погрешность
    Extended Eps; // Относительная погрешность
    Extended RealEps; // Реальная точность, полученная после вычисления
    TIntegralMethod IntMeth; // Метод интегрирования
    int iErr; // Индекс ошибки
    __fastcall TIntPar();
    AnsiString __fastcall GetErrStr();
};
//---------------------------------------------------------------------------
Extended __fastcall Integral(TIntPar *IntPar);
Extended __fastcall IntMethRect(HUnderIntFunc hf, const Extended& a,\
            const Extended& b, const Extended& abs, const Extended& eps,\
            int& iErr, Extended& Real);
Extended __fastcall IntMethTrap(HUnderIntFunc hf, const Extended& a,\
            const Extended& b, const Extended& abs, const Extended& eps,\
            int& iErr, Extended& Real);
Extended __fastcall IntMethSimp(HUnderIntFunc hf, const Extended& a,\
            const Extended& b, const Extended& abs, const Extended& eps,\
            int& iErr, Extended& Real);
//---------------------------------------------------------------------------
#endif
