//---------------------------------------------------------------------------
#ifndef integralH
#define integralH
//---------------------------------------------------------------------------
/* Integration method:
    miRectangle - rectangle method
    miTrapezoid - trapezoid method
    miSimpson   - Simpson's method
*/
enum TIntegralMethod { miRectangle, miTrapezoid, miSimpson };
//---------------------------------------------------------------------------
/* Type for denoting the integrand function */
typedef Extended __fastcall (*HUnderIntFunc)(Extended);
//---------------------------------------------------------------------------
/* Structure for passing parameters to the integration function */
struct TIntPar {
    HUnderIntFunc F;         // Integrand function
    Extended A, B;           // Integration limits
    Extended Abs;            // Absolute error
    Extended Eps;            // Relative error
    Extended RealEps;        // Actual accuracy obtained after calculation
    TIntegralMethod IntMeth; // Integration method
    int iErr;                // Error index
    __fastcall TIntPar();
    AnsiString __fastcall GetErrStr();
};
//---------------------------------------------------------------------------
Extended __fastcall Integral(TIntPar* IntPar);
Extended __fastcall IntMethRect(HUnderIntFunc hf, const Extended& a, const Extended& b, const Extended& abs,
                                const Extended& eps, int& iErr, Extended& Real);
Extended __fastcall IntMethTrap(HUnderIntFunc hf, const Extended& a, const Extended& b, const Extended& abs,
                                const Extended& eps, int& iErr, Extended& Real);
Extended __fastcall IntMethSimp(HUnderIntFunc hf, const Extended& a, const Extended& b, const Extended& abs,
                                const Extended& eps, int& iErr, Extended& Real);
//---------------------------------------------------------------------------
#endif
