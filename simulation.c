#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include <omp.h>
#include <immintrin.h>

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))


/* Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions (ie no particle moves more than one cell width in one
 * timestep). Otherwise the simulation becomes unstable.
 */
void setTimestepInterval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau)
{
    int i, j;
    float umax, vmax, deltu, deltv, deltRe;

    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        umax = 1.0e-10;
        vmax = 1.0e-10;
        for (i=0; i<=imax+1; i++) {
            for (j=1; j<=jmax+1; j++) {
                umax = max(fabs(u[i][j]), umax);
            }
        }
        for (i=1; i<=imax+1; i++) {
            for (j=0; j<=jmax+1; j++) {
                vmax = max(fabs(v[i][j]), vmax);
            }
        }

        deltu = delx/umax;
        deltv = dely/vmax;
        deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;

        if (deltu<deltv) {
            *del_t = min(deltu, deltRe);
        } else {
            *del_t = min(deltv, deltRe);
        }
        *del_t = tau * (*del_t); /* multiply by safety factor */
    }
}


/* Computation of tentative velocity field (f, g) */
void computeTentativeVelocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re)
{
    int  i=0;
    int j=0;
    float du2dx = 0.0;
    float duvdx = 0.0;
    float laplu = 0.0;
    float dv2dy = 0.0;
    float duvdy = 0.0;
    float laplv = 0.0;

    /* unrolled edges from loop to allow vectors to access previous indexes in array
       (no u[-1][j] conditions, for example) */
    for (i=0; i<=imax; i++) {
        f[i][0] = u[i][0];
    }
    for (j=0; j<=jmax; j++) {
        f[0][j] = u[0][j];
    }

    /* initialise vector placeholders */
    __m128 one;
    __m128 two;
    __m128 three;
    __m128 four;
    __m128 five;
    __m128 six;
    __m128 seven;
    __m128 eight;
    __m128 nine;
    __m128 ten;
    __m128 eleven;
    __m128 twelve;
    __m128 fabs_one;
    __m128 fabs_five;
    __m128 fabs_six;

    /* initialise vectors for catching current iteration values for u and v */
    __m128 au;
    __m128 bu;
    __m128 cu;
    __m128 du;
    __m128 eu;
    __m128 fu;
    __m128 av;
    __m128 bv;
    __m128 cv;
    __m128 dv;
    __m128 ev;
    __m128 fv;

    /* initialise vectors for f an g, and variables for calculation */
    //holds value for f when if condition is true
    __m128 true_vec_f;
    //holds value for f when if condition is false
    __m128 false_vec_f;
    //holds value for g when if condition is true
    __m128 true_vec_g;
    //holds value for g when if condition is false
    __m128 false_vec_g;
    __m128 result;
    __m128 vec_du2dx;
    __m128 vec_duvdy;
    __m128 vec_laplu;
    __m128 vec_duvdx;
    __m128 vec_dv2dy;
    __m128 vec_laplv;
    // holds mask to replace need for if statements
    __m128 mask;

    //gamma
    __m128 vec_gamma = _mm_set1_ps(gamma);
    //delx
    __m128 vec_delx = _mm_set1_ps(delx);
    //dely
    __m128 vec_dely = _mm_set1_ps(dely);
    //del_t
    __m128 vec_del_t = _mm_set1_ps(del_t);
    //Re
    __m128 vec_Re = _mm_set1_ps(Re);


    #pragma omp parallel for private(j)
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j+=4) {

            //loads u[i][j], u[i][j+1], u[i][j+2], u[i][j+3]
            au = _mm_loadu_ps(u[i] + j);
            //loads u[i+1][j], u[i+1][j+1], u[i+1][j+2], u[i+1][j+3]
            bu = _mm_loadu_ps(u[i+1] + j);
            //loads u[i-1][j], u[i-1][j+1], u[i-1][j+2], u[i-1][j+3]
            cu = _mm_loadu_ps(u[i-1] + j);
            //loads u[i][j+1], u[i][j+2], u[i][j+3], u[i][j+4]
            du = _mm_loadu_ps(u[i] + 1 + j);
            //loads u[i][j-1], u[i][j], u[i][j+1], u[i][j+2]
            eu = _mm_loadu_ps(u[i] - 1 + j);
            //loads u[i-1][j+1], u[i-1][j+2], u[i-1][j+3], u[i-1][j+4]
            fu = _mm_loadu_ps(u[i-1] + 1 + j);
            //loads v[i][j], v[i][j+1], v[i][j+2], v[i][j+3]
            av = _mm_loadu_ps(v[i] + j);
            //loads v[i+1][j], v[i+1][j+1], v[i+1][j+2], v[i+1][j+3]
            bv = _mm_loadu_ps(v[i+1] + j);
            //loads v[i][j-1], v[i][j], v[i][j+1], v[i][j+2]
            cv = _mm_loadu_ps(v[i] - 1 + j);
            //loads v[i+1][j-1], v[i+1][j], v[i+1][j+1], v[i+1][j+2]
            dv = _mm_loadu_ps(v[i+1] - 1 + j);
            //loads v[i-1][j], v[i-1][j+1], v[i-1][j+2], v[i-1][j+3]
            ev = _mm_loadu_ps(v[i-1] + j);
            //loads v[i][j+1], v[i][j+2], v[i][j+3], v[i][j+4]
            fv = _mm_loadu_ps(v[i] + 1 + j);

            // ((flag[i][j] & C_F) && (flag[i+1][j] & C_F))
            mask = _mm_setr_ps(((flag[i][j] & C_F) && (flag[i+1][j] & C_F)),
                               ((flag[i][j+1] & C_F) && (flag[i+1][j+1] & C_F)),
                               ((flag[i][j+2] & C_F) && (flag[i+1][j+2] & C_F)),
                               ((flag[i][j+3] & C_F) && (flag[i+1][j+3] & C_F)));

            //fixes issue where 1.0 represented true, now -nan does
            mask = _mm_cmpeq_ps(mask, _mm_set1_ps(1.0));

            // (u[i][j]+u[i+1][j])
            one = _mm_add_ps(au, bu);
            // fabs((u[i][j]+u[i+1][j]))
            fabs_one = _mm_sqrt_ps(_mm_mul_ps(one, one));
            // (u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])
            two = _mm_mul_ps(one, one);
            // (u[i][j]-u[i+1][j])
            three = _mm_sub_ps(au, bu);
            // gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])
            four = _mm_mul_ps(_mm_mul_ps(vec_gamma, fabs_one), three);
            // (u[i-1][j]+u[i][j])
            five = _mm_add_ps(cu, au);
            // fabs(u[i][j]+u[i+1][j])
            fabs_five = _mm_sqrt_ps(_mm_mul_ps(five, five));
            // (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])
            six = _mm_mul_ps(five, five);
            // (u[i-1][j]-u[i][j])
            seven = _mm_sub_ps(cu, au);
            // gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j])
            eight = _mm_mul_ps(_mm_mul_ps(vec_gamma, fabs_five), seven);
            // eq. before division
            nine = _mm_sub_ps(_mm_add_ps(two, four), _mm_add_ps(six, eight));
            // (4.0*delx)
            ten = _mm_mul_ps(_mm_set1_ps(4.0), vec_delx);
            // du2dx
            vec_du2dx = _mm_div_ps(nine, ten);


            // (v[i][j]+v[i+1][j])
            one = _mm_add_ps(av, bv);
            // fabs(v[i][j]+v[i+1][j])
            fabs_one = _mm_sqrt_ps(_mm_mul_ps(one, one));
            // (u[i][j]+u[i][j+1])
            two = _mm_add_ps(au, du);
            // (v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])
            three = _mm_mul_ps(one, two);
            // (u[i][j]-u[i][j+1])
            four = _mm_sub_ps(au, du);
            // gamma*fabs(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])
            five = _mm_mul_ps(_mm_mul_ps(vec_gamma, fabs_one), four);
            // (v[i][j-1]+v[i+1][j-1])
            six = _mm_add_ps(cv, dv);
            // fabs(v[i][j-1]+v[i+1][j-1])
            fabs_six = _mm_sqrt_ps(_mm_mul_ps(six, six));
            // (u[i][j-1]+u[i][j])
            seven = _mm_add_ps(au, eu);
            // (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])
            eight = _mm_mul_ps(six, seven);
            // (u[i][j-1]-u[i][j])
            nine = _mm_sub_ps(eu, au);
            // gamma*fabs(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j])
            ten = _mm_mul_ps(_mm_mul_ps(vec_gamma, fabs_six), nine);
            // eq. before division
            eleven = _mm_sub_ps(_mm_add_ps(three, five), _mm_add_ps(eight, ten));
            // (4.0*dely)
            twelve = _mm_mul_ps(_mm_set1_ps(4.0), vec_dely);
            // duvdy
            vec_duvdy = _mm_div_ps(eleven, twelve);


            // (u[i+1][j]-2.0*u[i][j]+u[i-1][j])
            one = _mm_add_ps(cu, _mm_sub_ps(bu, _mm_mul_ps(_mm_set1_ps(2.0), au)));
            // (u[i+1][j]-2.0*u[i][j]+u[i-1][j])/delx
            two = _mm_div_ps(one, vec_delx);
            // (u[i+1][j]-2.0*u[i][j]+u[i-1][j])/delx/delx
            three = _mm_div_ps(two, vec_delx);
            // (u[i][j+1]-2.0*u[i][j]+u[i][j-1])
            four = _mm_add_ps(eu, _mm_sub_ps(du, _mm_mul_ps(_mm_set1_ps(2.0), au)));
            // (u[i][j+1]-2.0*u[i][j]+u[i][j-1])/dely
            five = _mm_div_ps(four, vec_dely);
            // (u[i][j+1]-2.0*u[i][j]+u[i][j-1])/dely/dely
            six = _mm_div_ps(five, vec_dely);
            // laplu
            vec_laplu = _mm_add_ps(three, six);


            // laplu/Re
            one = _mm_div_ps(vec_laplu, vec_Re);
            // (laplu/Re-du2dx-duvdy)
            two = _mm_sub_ps(one, _mm_add_ps(vec_du2dx, vec_duvdy));
            // del_t*(laplu/Re-du2dx-duvdy)
            three = _mm_mul_ps(vec_del_t, two);
            // u[i][j]+del_t*(laplu/Re-du2dx-duvdy)
            true_vec_f = _mm_add_ps(au, three);

            // f = u[i][j]
            false_vec_f = au;

            // result for fs
            result = _mm_or_ps(_mm_and_ps(mask, true_vec_f), _mm_andnot_ps(mask, false_vec_f));
            // store into f
            _mm_storeu_ps(f[i]+j, result);
            // f[i][j]= temp[0];


            // ((flag[i][j] & C_F) && (flag[i][j+1] & C_F))
            mask = _mm_setr_ps(((flag[i][j] & C_F) && (flag[i][j+1] & C_F)),
                              ((flag[i][j+1] & C_F) && (flag[i][j+2] & C_F)),
                              ((flag[i][j+2] & C_F) && (flag[i][j+3] & C_F)),
                              ((flag[i][j+3] & C_F) && (flag[i][j+4] & C_F)));

            //fixes issue where 1.0 represented true, now -nan does
            mask = _mm_cmpeq_ps(mask, _mm_set1_ps(1.0));

            // (u[i][j]+u[i][j+1])
            one = _mm_add_ps(au, du);
            // fabs(u[i][j]+u[i][j+1])
            fabs_one = _mm_sqrt_ps(_mm_mul_ps(one, one));
            // (v[i][j]+v[i+1][j])
            two = _mm_add_ps(av, bv);
            // (u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])
            three = _mm_mul_ps(one, two);
            // (v[i][j]-v[i+1][j])
            four = _mm_sub_ps(av, bv);
            // gamma*fabs(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])
            five = _mm_mul_ps(_mm_mul_ps(vec_gamma, fabs_one), four);
            // (u[i-1][j]+u[i-1][j+1])
            six = _mm_add_ps(cu, fu);
            // fabs(u[i-1][j]+u[i-1][j+1])
            fabs_six = _mm_sqrt_ps(_mm_mul_ps(six, six));
            // (v[i-1][j]+v[i][j])
            seven = _mm_add_ps(ev, av);
            // (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])
            eight = _mm_mul_ps(six, seven);
            // (v[i-1][j]-v[i][j])
            nine = _mm_sub_ps(ev, av);
            // gamma*fabs(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j])
            ten = _mm_mul_ps(_mm_mul_ps(vec_gamma, fabs_six), nine);
            // eq. before division
            eleven = _mm_sub_ps(_mm_add_ps(three, five), _mm_add_ps(eight, ten));
            // (4.0*delx)
            twelve = _mm_mul_ps(_mm_set1_ps(4.0), vec_delx);
            // duvdx
            vec_duvdx = _mm_div_ps(eleven, twelve);


            // (v[i][j]+v[i][j+1])
            one = _mm_add_ps(av, fv);
            // fabs(v[i][j]+v[i][j+1])
            fabs_one = _mm_sqrt_ps(_mm_mul_ps(one, one));
            // (v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])
            two = _mm_mul_ps(one, one);
            // (v[i][j]-v[i][j+1])
            three = _mm_sub_ps(av, fv);
            // gamma*fabs(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])
            four = _mm_mul_ps(_mm_mul_ps(vec_gamma, fabs_one), three);
            // (v[i][j-1]+v[i][j])
            five = _mm_add_ps(cv, av);
            // fabs(v[i][j-1]+v[i][j])
            fabs_five = _mm_sqrt_ps(_mm_mul_ps(five, five));
            // (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])
            six = _mm_mul_ps(five, five);
            // (v[i][j-1]-v[i][j])
            seven = _mm_sub_ps(cv, av);
            // gamma*fabs(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j])
            eight = _mm_mul_ps(_mm_mul_ps(vec_gamma, fabs_five), seven);
            // eq. before division
            nine = _mm_sub_ps(_mm_add_ps(two, four), _mm_add_ps(six, eight));
            // (4.0*dely)
            ten = _mm_mul_ps(_mm_set1_ps(4.0), vec_dely);
            // dv2dy
            vec_dv2dy = _mm_div_ps(nine, ten);


            // (v[i+1][j]-2.0*v[i][j]+v[i-1][j])
            one = _mm_add_ps(ev, _mm_sub_ps(bv, _mm_mul_ps(_mm_set1_ps(2.0), av)));
            // (v[i+1][j]-2.0*v[i][j]+v[i-1][j])/delx
            two = _mm_div_ps(one, vec_delx);
            // (v[i+1][j]-2.0*v[i][j]+v[i-1][j])/delx/delx
            three = _mm_div_ps(two, vec_delx);
            // (v[i][j+1]-2.0*v[i][j]+v[i][j-1])
            four = _mm_add_ps(cv, _mm_sub_ps(fv, _mm_mul_ps(_mm_set1_ps(2.0), av)));
            // (v[i][j+1]-2.0*v[i][j]+v[i][j-1])/dely
            five = _mm_div_ps(four, vec_dely);
            // (v[i][j+1]-2.0*v[i][j]+v[i][j-1])/dely/dely
            six = _mm_div_ps(five, vec_dely);
            // laplv
            vec_laplv = _mm_add_ps(three, six);


            // laplv/Re
            one = _mm_div_ps(vec_laplv, vec_Re);
            // (laplv/Re-duvdx-dv2dy)
            two = _mm_sub_ps(one, _mm_add_ps(vec_dv2dy, vec_duvdx));
            // del_t*(laplv/Re-duvdx-dv2dy)
            three = _mm_mul_ps(vec_del_t, two);
            // v[i][j]+del_t*(laplv/Re-duvdx-dv2dy)
            true_vec_g = _mm_add_ps(av, three);

            // g = v[i][j]
            false_vec_g = av;

            // result for gs
            result = _mm_or_ps(_mm_and_ps(mask, true_vec_g), _mm_andnot_ps(mask, false_vec_g));

            // store into g
            _mm_storeu_ps(g[i]+j, result);
        }
    }
}


/* Calculate the right hand side of the pressure equation */
void computeRhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely)
{
    int i, j;

    // placeholdsers for vector operations
    __m128 one;
    __m128 two;
    __m128 three;
    __m128 four;
    __m128 five;

    __m128 vec_delx = _mm_set1_ps(delx);
    __m128 vec_dely = _mm_set1_ps(dely);
    __m128 vec_del_t = _mm_set1_ps(del_t);
    __m128 mask;
    __m128 af;
    __m128 bf;
    __m128 ag;
    __m128 bg;
    __m128 vec_rhs;

    #pragma omp parallel for private(j, vec_rhs, af, bf, ag, bg, mask)
    for (i=1;i<=imax;i++) {
        __m128 result;
        for (j=1;j<=jmax;j+=4) {
            //loads rhs[i][j], rhs[i][j+1], rhs[i][j+2], rhs[i][j+3]
            vec_rhs = _mm_loadu_ps(rhs[i] + j);
            // loads f[i][j], f[i][j+1], f[i][j+2], f[i][j+3]
            af = _mm_loadu_ps(f[i] + j);
            // loads f[i-1][j], f[i-1][j+1], f[i-1][j+2], f[i-1][j+3]
            bf = _mm_loadu_ps(f[i-1] + j);
            // loads g[i][j], g[i][j+1], g[i][j+2], g[i][j+3]
            ag = _mm_loadu_ps(g[i] + j);
            // loads g[i][j-1], g[i][j], g[i][j+1], g[i][j+2]
            bg = _mm_loadu_ps(g[i] -1 + j);

            // (flag[i][j] & C_F)
            mask = _mm_setr_ps((flag[i][j] & C_F),
                               (flag[i][j+1] & C_F),
                               (flag[i][j+2] & C_F),
                               (flag[i][j+3] & C_F));

            //fixes issue where 1.0 represented true, now -nan does
            mask = _mm_cmpeq_ps(mask, _mm_set1_ps(16.0));

            // (f[i][j]-f[i-1][j])
            one = _mm_sub_ps(af, bf);
            // (f[i][j]-f[i-1][j])/delx
            two = _mm_div_ps(one, vec_delx);
            // (g[i][j]-g[i][j-1])
            three = _mm_sub_ps(ag, bg);
            // (g[i][j]-g[i][j-1])/dely
            four = _mm_div_ps(three, vec_dely);
            // (f[i][j]-f[i-1][j])/delx + (g[i][j]-g[i][j-1])/dely
            five = _mm_add_ps(two, four);

            // correct values for p
            result = _mm_or_ps(_mm_and_ps(mask, _mm_div_ps(five, vec_del_t)), _mm_andnot_ps(mask, vec_rhs));
            _mm_storeu_ps(rhs[i]+j, result);
        }
    }
}


/* Red/Black SOR to solve the poisson equation */
int poissonSolver(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull)
{
    int i, j, iter;
    // float add;
    float beta_mod;
    // float p0 = 0.0;
    int rb; /* Red-black value. */

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);

    //temporary vector stores for vector calculations
    __m128 one;
    __m128 two;
    __m128 three;
    __m128 four;
    __m128 five;
    __m128 six;
    __m128 seven;
    __m128 eight;
    __m128 nine;
    __m128 ten;
    __m128 eleven;
    __m128 twelve;
    __m128 thirteen;

    //needed since deals with array values too spread apart to place immediately
    float temp_store[4];

    //holds which values in vector can have new p applied
    __m128 mask;
    // loads -omega
    __m128 vec_omega = _mm_set1_ps(-omega);
    // loads rdy2
    __m128 vec_rdy2 = _mm_set1_ps(rdy2);
    // loads rdx2
    __m128 vec_rdx2 = _mm_set1_ps(rdx2);
    // holds beta_mod
    __m128 vec_beta_mod;
    // holds new p values
    __m128 new_p;

    //stores for different levels of p
    __m128 ap;
    __m128 bp;
    __m128 cp;
    __m128 dp;
    __m128 ep;
    // store for rhs of current iterations
    __m128 vec_rhs;
    // eps_N values
    __m128 vec_eps_N;
    // eps_S values
    __m128 vec_eps_S;
    // eps_E values
    __m128 vec_eps_E;
    // eps_W values
    __m128 vec_eps_W;


    /* Red/Black SOR-iteration */
    for (iter = 0; iter < itermax; iter++) {
        for (rb = 0; rb <= 1; rb++) {
            #pragma omp parallel for private(j, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, temp_store, mask, vec_beta_mod, new_p, ap, bp, cp, dp, ep, vec_rhs, vec_eps_N, vec_eps_S, vec_eps_E, vec_eps_W)
            for (i = 1; i <= imax; i++) {
                //extra vector store
                __m128 result;
                // if i is odd and rb = 0 or i is even and rb = 1, all odd j's
                if (((i % 2 == 1) && (rb == 0)) || ((i % 2 == 0) && (rb == 1))) { j = 1; }
                // if i is odd and rb = 1 or i is even and rb = 0, all even j's
                else { j = 2; }
                for (; j <= jmax; j+=8) {

                    // (flag[i][j] & C_F)
                    mask = _mm_setr_ps((flag[i][j] & C_F),
                                       (flag[i][j+2] & C_F),
                                       (flag[i][j+4] & C_F),
                                       (flag[i][j+6] & C_F));

                    //fixes issue where 1.0 represented true, now -nan does
                    mask = _mm_cmpeq_ps(mask, _mm_set1_ps(16.0));

                    // loads  p[i][j], p[i][j+2], p[i][j+4], p[i][j+6]
                    ap = _mm_shuffle_ps(_mm_loadu_ps(p[i] + j), _mm_loadu_ps(p[i] + j + 4), _MM_SHUFFLE(2,0,2,0));
                    // bp = _mm_loadu_ps(p[i+1] + j);
                    // loads p[i+1][j], p[i+1][j+2], p[i+1][j+4], p[i+1][j+6]
                    bp = _mm_shuffle_ps(_mm_loadu_ps(p[i+1] + j), _mm_loadu_ps(p[i+1] + j + 4), _MM_SHUFFLE(2,0,2,0));
                    // loads p[i-1][j], p[i-1][j+2], p[i-1][j+4], p[i-1][j+6]
                    cp = _mm_shuffle_ps(_mm_loadu_ps(p[i-1] + j), _mm_loadu_ps(p[i-1] + j + 4), _MM_SHUFFLE(2,0,2,0));
                    // loads p[i][j+1], p[i][j+3], p[i][j+5], p[i][j+7]
                    dp = _mm_shuffle_ps(_mm_loadu_ps(p[i] + 1 + j), _mm_loadu_ps(p[i] + 1 + j + 4), _MM_SHUFFLE(2,0,2,0));
                    // loads p[i][j-1], p[i][j+1], p[i][j+3], p[i][j+5]
                    ep = _mm_shuffle_ps(_mm_loadu_ps(p[i] - 1 + j), _mm_loadu_ps(p[i] - 1 + j + 4), _MM_SHUFFLE(2,0,2,0));
                    // loads rhs[i][j], rhs[i][j+2], rhs[i][j+4], rhs[i][j+8]
                    vec_rhs = _mm_shuffle_ps(_mm_loadu_ps(rhs[i] + j), _mm_loadu_ps(rhs[i] + j + 4), _MM_SHUFFLE(2,0,2,0));
                    // loads eps_N
                    vec_eps_N = _mm_div_ps(_mm_setr_ps((float) (flag[i][j+1] & C_F), (float) (flag[i][j+3] & C_F), (float) (flag[i][j+5] & C_F), (float) (flag[i][j+7] & C_F)), _mm_set1_ps(16.0));
                    // loads eps_S
                    vec_eps_S = _mm_div_ps(_mm_setr_ps((float) (flag[i][j-1] & C_F), (float) (flag[i][j+1] & C_F), (float) (flag[i][j+3] & C_F), (float) (flag[i][j+5] & C_F)), _mm_set1_ps(16.0));
                    // loads eps_E
                    vec_eps_E = _mm_div_ps(_mm_setr_ps((float) (flag[i+1][j] & C_F), (float) (flag[i+1][j+2] & C_F), (float) (flag[i+1][j+4] & C_F), (float) (flag[i+1][j+6] & C_F)), _mm_set1_ps(16.0));
                    // loads eps_W
                    vec_eps_W = _mm_div_ps(_mm_setr_ps((float) (flag[i-1][j] & C_F), (float) (flag[i-1][j+2] & C_F), (float) (flag[i-1][j+4] & C_F), (float) (flag[i-1][j+6] & C_F)), _mm_set1_ps(16.0));


                    // (eps_E+eps_W)
                    one = _mm_add_ps(vec_eps_E, vec_eps_W);
                    // (eps_E+eps_W)*rdx2
                    two = _mm_mul_ps(one, vec_rdx2);
                    // (eps_N+eps_S)
                    three = _mm_add_ps(vec_eps_N, vec_eps_S);
                    // (eps_N+eps_S)*rdy2
                    four = _mm_mul_ps(three, vec_rdy2);
                    // ((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2)
                    five = _mm_add_ps(two, four);
                    // -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2)
                    vec_beta_mod = _mm_div_ps(vec_omega, five);


                    // (1.-omega)
                    one = _mm_add_ps(vec_omega, _mm_set1_ps(1.0));
                    // (1.-omega)*p[i][j]
                    two = _mm_mul_ps(one, ap);
                    // eps_E*p[i+1][j]
                    three = _mm_mul_ps(vec_eps_E, bp);
                    // eps_W*p[i-1][j]
                    four = _mm_mul_ps(vec_eps_W, cp);
                    // (eps_E*p[i+1][j]+eps_W*p[i-1][j])
                    five = _mm_add_ps(three, four);
                    // (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2
                    six = _mm_mul_ps(five, vec_rdx2);
                    // eps_N*p[i][j+1]
                    seven = _mm_mul_ps(vec_eps_N, dp);
                    // eps_S*p[i][j-1]
                    eight = _mm_mul_ps(vec_eps_S, ep);
                    // (eps_N*p[i][j+1]+eps_S*p[i][j-1])
                    nine = _mm_add_ps(seven, eight);
                    // (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                    ten = _mm_mul_ps(nine, vec_rdy2);
                    // (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2 + (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                    eleven = _mm_add_ps(six, ten);
                    // factor to multiply by beta_mod
                    twelve = _mm_sub_ps(eleven, vec_rhs);
                    // thing to take from (1.-omega)*p[i][j]
                    thirteen = _mm_mul_ps(vec_beta_mod, twelve);
                    // new p
                    new_p = _mm_sub_ps(two, thirteen);


                    // correct values for p
                    result = _mm_or_ps(_mm_and_ps(mask, new_p), _mm_andnot_ps(mask, ap));
                    _mm_storeu_ps(temp_store, result);

                    //store spaced-out values
                    p[i][j] = temp_store[0];
                    p[i][j+2] = temp_store[1];
                    p[i][j+4] = temp_store[2];
                    p[i][j+6] = temp_store[3];

                } /* end of j */
            } /* end of i */
          /* end of parallel section */
        } /* end of rb */
        /* res did not affect the program but intrinsics were added, so just commented out to keep example of vector intrinsics*/
        // //create temporary non-address based var to hold residual
        // float temp_res = 0.0;
        // #pragma omp parallel for private(j, mask, ap, bp, cp, dp, ep, vec_rhs, vec_eps_N, vec_eps_S, vec_eps_E, vec_eps_W, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen) reduction(+:temp_res, p0)
        // for (i = 1; i <= imax; i++) {
        //     //extra vectors for operations
        //     __m128 vec_p0;
        //     __m128 result;
        //     __m128 sum;
        //     __m128 vec_add;
        //     __m128 vec_temp_res;
        //
        //     for (j = 1; j+4 <= jmax-1; j+=4) {
        //
        //         // (flag[i][j] & C_F)
        //         mask = _mm_setr_ps((flag[i][j] & C_F),
        //                            (flag[i][j+1] & C_F),
        //                            (flag[i][j+2] & C_F),
        //                            (flag[i][j+3] & C_F));
        //
        //         //fixes issue where 1.0 represented true, now -nan does
        //         mask = _mm_cmpeq_ps(mask, _mm_set1_ps(16.0));
        //
        //         // loads p[i][j], p[i][j+1], p[i][j+2], p[i][j+3],
        //         ap = _mm_loadu_ps(p[i] + j);
        //         // loads p[i+1][j], p[i+1][j+1], p[i+1][j+2], p[i+1][j+3],
        //         bp = _mm_loadu_ps(p[i+1] + j);
        //         // loads p[i-1][j], p[i-1][j+1], p[i-1][j+2], p[i-1][j+3],
        //         cp = _mm_loadu_ps(p[i-1] + j);
        //         // loads p[i][j+1], p[i][j+2], p[i][j+3], p[i][j+4],
        //         dp = _mm_loadu_ps(p[i] + j + 1);
        //         // loads p[i][j-1], p[i][j], p[i][j+1], p[i][j+2],
        //         ep = _mm_loadu_ps(p[i] + j - 1);
        //         // loads rhs[i][j], rhs[i][j+1], rhs[i][j+2], rhs[i][j+3]
        //         vec_rhs = _mm_loadu_ps(rhs[i] + j);
        //         // loads eps_N
        //         vec_eps_N = _mm_div_ps(_mm_setr_ps((float) (flag[i][j+1] & C_F), (float) (flag[i][j+2] & C_F), (float) (flag[i][j+3] & C_F), (float) (flag[i][j+4] & C_F)), _mm_set1_ps(16.0));
        //         // loads eps_S
        //         vec_eps_S = _mm_div_ps(_mm_setr_ps((float) (flag[i][j-1] & C_F), (float) (flag[i][j] & C_F), (float) (flag[i][j+1] & C_F), (float) (flag[i][j+2] & C_F)), _mm_set1_ps(16.0));
        //         // loads eps_E
        //         vec_eps_E = _mm_div_ps(_mm_setr_ps((float) (flag[i+1][j] & C_F), (float) (flag[i+1][j+1] & C_F), (float) (flag[i+1][j+2] & C_F), (float) (flag[i+1][j+3] & C_F)), _mm_set1_ps(16.0));
        //         // loads eps_W
        //         vec_eps_W = _mm_div_ps(_mm_setr_ps((float) (flag[i-1][j] & C_F), (float) (flag[i-1][j+1] & C_F), (float) (flag[i-1][j+2] & C_F), (float) (flag[i-1][j+3] & C_F)), _mm_set1_ps(16.0));
        //
        //         // p[i][j]*p[i][j]
        //         vec_p0 = _mm_mul_ps(ap, ap);
        //         //find which ones are valid for adding to p0 sum
        //         result = _mm_or_ps(_mm_and_ps(mask, vec_p0), _mm_andnot_ps(mask, _mm_set1_ps(0.0)));
        //         //running hadd twice makes every element the sum of everything in vector
        //         sum = _mm_hadd_ps(result, result);
        //         sum = _mm_hadd_ps(sum, sum);
        //         //get sum from vector
        //         _mm_storeu_ps(temp_store, sum);
        //         p0 += temp_store[0];
        //
        //
        //         // (p[i+1][j]-p[i][j])
        //         one = _mm_sub_ps(bp, ap);
        //         // eps_E*(p[i+1][j]-p[i][j])
        //         two = _mm_mul_ps(vec_eps_E, one);
        //         // (p[i][j]-p[i-1][j])
        //         three = _mm_sub_ps(ap, cp);
        //         // eps_W*(p[i][j]-p[i-1][j])
        //         four = _mm_mul_ps(vec_eps_W, three);
        //         // (eps_E*(p[i+1][j]-p[i][j]) - eps_W*(p[i][j]-p[i-1][j]))
        //         five = _mm_sub_ps(two, four);
        //         // (eps_E*(p[i+1][j]-p[i][j]) - eps_W*(p[i][j]-p[i-1][j])) * rdx2
        //         six = _mm_mul_ps(five, vec_rdx2);
        //         // (p[i][j+1]-p[i][j])
        //         seven = _mm_sub_ps(dp, ap);
        //         // eps_N*(p[i][j+1]-p[i][j])
        //         eight = _mm_mul_ps(vec_eps_N, seven);
        //         // (p[i][j]-p[i][j-1])
        //         nine = _mm_sub_ps(ap, ep);
        //         // eps_S*(p[i][j]-p[i][j-1])
        //         ten = _mm_mul_ps(vec_eps_S, nine);
        //         // (eps_N*(p[i][j+1]-p[i][j]) - eps_S*(p[i][j]-p[i][j-1]))
        //         eleven = _mm_sub_ps(eight, ten);
        //         // (eps_N*(p[i][j+1]-p[i][j]) - eps_S*(p[i][j]-p[i][j-1])) * rdy2
        //         twelve = _mm_mul_ps(eleven, vec_rdy2);
        //         // (eps_E*(p[i+1][j]-p[i][j]) - eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
        //         // (eps_N*(p[i][j+1]-p[i][j]) - eps_S*(p[i][j]-p[i][j-1])) * rdy2
        //         thirteen = _mm_add_ps(six, twelve);
        //         // add
        //         vec_add = _mm_sub_ps(thirteen, vec_rhs);
        //         // holds temp_res from the 4 array elements
        //         vec_temp_res = _mm_mul_ps(vec_add, vec_add);
        //         //find which ones are valid for adding to temp_res sum
        //         result = _mm_or_ps(_mm_and_ps(mask, vec_temp_res), _mm_andnot_ps(mask, _mm_set1_ps(0.0)));
        //         _mm_storeu_ps(temp_store, result);
        //         //sum up the values in vector
        //         sum = _mm_hadd_ps(result, result);
        //         sum = _mm_hadd_ps(sum, sum);
        //         //get sum from vector
        //         _mm_storeu_ps(temp_store, sum);
        //         temp_res += temp_store[0];
        //     }
        //     //catch the rest
        //     for (; j<=jmax; j++) {
        //         if (flag[i][j] & C_F) {
        //             /* moved here from fusing computing sum of squares */
        //             p0 += p[i][j]*p[i][j];
        //             /* only fluid cells */
        //             add = (eps_E*(p[i+1][j]-p[i][j]) -
        //                 eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
        //                 (eps_N*(p[i][j+1]-p[i][j]) -
        //                 eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
        //             temp_res += add*add;
        //         }
        //     }
        // }
        // /* more from dispursing sum of squares loop */
        // p0 = sqrt(p0/ifull);
        // if (p0 < 0.0001) { p0 = 1.0; }
        // *res = sqrt((temp_res)/ifull)/p0;
        //
        // /* convergence? */
        // if (*res<eps) break;
    } /* end of iter */

    return iter;
}


/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void updateVelocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely)
{
    int i, j;

    //temporary vector stores for vector calculations
    __m128 one;
    __m128 two;

    __m128 vec_u;
    __m128 vec_v;
    __m128 vec_f;
    __m128 vec_g;
    __m128 ap;
    __m128 bp;
    __m128 cp;
    __m128 mask;
    __m128 vec_delx = _mm_set1_ps(delx);
    __m128 vec_dely = _mm_set1_ps(dely);
    __m128 vec_del_t = _mm_set1_ps(del_t);
    // del_t/delx
    __m128 vec_t_by_x = _mm_div_ps(vec_del_t, vec_delx);
    // del_t/dely
    __m128 vec_t_by_y = _mm_div_ps(vec_del_t, vec_dely);

    #pragma omp parallel for private(j, one, two, vec_u, vec_v, vec_f, vec_g, ap, bp, cp)
    for (i=1; i<=imax; i++) {
        __m128 result;
        for (j=1; j<=jmax; j+=4) {
            // loads u[i][j], u[i][j+1], u[i][j+2], u[i][j+3]
            vec_u = _mm_loadu_ps(u[i] + j);
            // loads v[i][j], v[i][j+1], v[i][j+2], v[i][j+3]
            vec_v = _mm_loadu_ps(v[i] + j);
            // loads f[i][j], f[i][j+1], f[i][j+2], f[i][j+3]
            vec_f = _mm_loadu_ps(f[i] + j);
            // loads g[i][j], g[i][j+1], g[i][j+2], g[i][j+3]
            vec_g = _mm_loadu_ps(g[i] + j);
            // loads p[i][j], p[i][j+1], p[i][j+2], p[i][j+3]
            ap =_mm_loadu_ps(p[i] + j);
            // loads p[i+1][j], p[i+1][j+1], p[i+1][j+2], p[i+1][j+3]
            bp =_mm_loadu_ps(p[i+1] + j);
            // loads p[i][j+1], p[i][j+2], p[i][j+3], p[i][j+4]
            cp = _mm_loadu_ps(p[i] + 1 + j);


            // ((i != imax) && (flag[i][j] & C_F) && (flag[i+1][j] & C_F))
            mask = _mm_setr_ps(((i != imax) && (flag[i][j] & C_F) && (flag[i+1][j] & C_F)),
                               ((i != imax) && (flag[i][j+1] & C_F) && (flag[i+1][j+1] & C_F)),
                               ((i != imax) && (flag[i][j+2] & C_F) && (flag[i+1][j+2] & C_F)),
                               ((i != imax) && (flag[i][j+3] & C_F) && (flag[i+1][j+3] & C_F)));

            //fixes issue where 1.0 represented true, now -nan does
            mask = _mm_cmpeq_ps(mask, _mm_set1_ps(1.0));

            // (p[i+1][j]-p[i][j])
            one = _mm_sub_ps(bp, ap);
            // (p[i+1][j]-p[i][j])*del_t/delx
            two = _mm_mul_ps(one, vec_t_by_x);
            // result for u
            result = _mm_or_ps(_mm_and_ps(mask, _mm_sub_ps(vec_f, two)), _mm_andnot_ps(mask, vec_u));
            _mm_storeu_ps(u[i]+j, result);


            // ((i != imax) && (flag[i][j] & C_F) && (flag[i+1][j] & C_F))
            mask = _mm_setr_ps(((j != jmax) && (flag[i][j] & C_F) && (flag[i][j+1] & C_F)),
                               ((j != jmax) && (flag[i][j+1] & C_F) && (flag[i][j+2] & C_F)),
                               ((j != jmax) && (flag[i][j+2] & C_F) && (flag[i][j+3] & C_F)),
                               ((j != jmax) && (flag[i][j+3] & C_F) && (flag[i][j+4] & C_F)));
           //fixes issue where 1.0 represented true, now -nan does
           mask = _mm_cmpeq_ps(mask, _mm_set1_ps(1.0));

            // (p[i][j+1]-p[i][j])
            one = _mm_sub_ps(cp, ap);
            // (p[i][j+1]-p[i][j])*del_t/dely
            two = _mm_mul_ps(one, vec_t_by_y);
            // result for u
            result = _mm_or_ps(_mm_and_ps(mask, _mm_sub_ps(vec_g, two)), _mm_andnot_ps(mask, vec_v));
            _mm_storeu_ps(v[i]+j, result);
        }
    }
}

void applyBoundaryConditions(float **u, float **v, char **flag,
    int imax, int jmax, float ui, float vi)
{
    int i, j;

    for (j=0; j<=jmax+1; j++) {
        /* Fluid freely flows in from the west */
        u[0][j] = u[1][j];
        v[0][j] = v[1][j];

        /* Fluid freely flows out to the east */
        u[imax][j] = u[imax-1][j];
        v[imax+1][j] = v[imax][j];
    }

    for (i=0; i<=imax+1; i++) {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        u[i][jmax+1] = u[i][jmax];
        u[i][0] = u[i][1];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */
    #pragma omp parallel for private(j)
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & B_NSEW) {
                switch (flag[i][j]) {
                    case B_N:
                        u[i][j]   = -u[i][j+1];
                        break;
                    case B_E:
                        v[i][j]   = -v[i+1][j];
                        break;
                    case B_S:
                        u[i][j]   = -u[i][j-1];
                        break;
                    case B_W:
                        v[i][j]   = -v[i-1][j];
                        break;
                    case B_NE:
                        break;
                    case B_SE:
                        v[i][j]   = -v[i+1][j];
                        break;
                    case B_SW:
                        v[i][j]   = -v[i-1][j];
                        u[i][j]   = -u[i][j-1];
                        break;
                    case B_NW:
                        u[i][j]   = -u[i][j+1];
                        break;
                }
            }
        }
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    v[0][0] = 2*vi-v[1][0];
    for (j=1;j<=jmax;j++) {
        u[0][j] = ui;
        v[0][j] = 2*vi-v[1][j];
    }
}
