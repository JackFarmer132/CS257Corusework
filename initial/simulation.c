#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include <immintrin.h>

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

/* Computation of tentative velocity field (f, g) */
void computeTentativeVelocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re)
{
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;

    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
                    gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
                    (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
                    gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
                    /(4.0*delx);
                duvdy = ((v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])+
                    gamma*fabs(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])-
                    (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])-
                    gamma*fabs(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j]))
                    /(4.0*dely);
                laplu = (u[i+1][j]-2.0*u[i][j]+u[i-1][j])/delx/delx+
                    (u[i][j+1]-2.0*u[i][j]+u[i][j-1])/dely/dely;

                f[i][j] = u[i][j]+del_t*(laplu/Re-du2dx-duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                duvdx = ((u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])+
                    gamma*fabs(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])-
                    (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])-
                    gamma*fabs(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j]))
                    /(4.0*delx);
                dv2dy = ((v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])+
                    gamma*fabs(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])-
                    (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])-
                    gamma*fabs(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j]))
                    /(4.0*dely);

                laplv = (v[i+1][j]-2.0*v[i][j]+v[i-1][j])/delx/delx+
                    (v[i][j+1]-2.0*v[i][j]+v[i][j-1])/dely/dely;

                g[i][j] = v[i][j]+del_t*(laplv/Re-duvdx-dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    /* f & g at external boundaries */
    for (j=1; j<=jmax; j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    for (i=1; i<=imax; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}


/* Calculate the right hand side of the pressure equation */
void computeRhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely)
{
    int i, j;

    for (i=1;i<=imax;i++) {
        for (j=1;j<=jmax;j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = (
                             (f[i][j]-f[i-1][j])/delx +
                             (g[i][j]-g[i][j-1])/dely
                            ) / del_t;
            }
        }
    }
}


/* Red/Black SOR to solve the poisson equation */
int poissonSolver(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull)
{
  int i, j, iter;
  float add, beta_2, beta_mod;
  float p0 = 0.0;
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
            // #pragma omp parallel for private(j, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, temp_store, mask, vec_beta_mod, new_p, ap, bp, cp, dp, ep, vec_rhs, vec_eps_N, vec_eps_S, vec_eps_E, vec_eps_W)
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

                // //need to catch the rest
                // for (; j<=jmax; j+=2) {
                //     if (flag[i][j] & C_F) {
                //         /* modified star near boundary */
                //         beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                //         p[i][j] = (1.-omega)*p[i][j] -
                //             beta_mod*(
                //                   (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2
                //                 + (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                //                 - rhs[i][j]
                //             );
                //     }
                // }
            } /* end of i */
          /* end of parallel section */
        } /* end of rb */
        /* res did not affect the program but intrinsics were added, so just commented out to keep example of vector intrinsics*/
        //create temporary non-address based var to hold residual
        float temp_res = 0.0;
        // #pragma omp parallel for private(j, mask, ap, bp, cp, dp, ep, vec_rhs, vec_eps_N, vec_eps_S, vec_eps_E, vec_eps_W, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen) reduction(+:temp_res, p0)
        for (i = 1; i <= imax; i++) {
            //extra vectors for operations
            __m128 vec_p0;
            __m128 result;
            __m128 sum;
            __m128 vec_add;
            __m128 vec_temp_res;

            for (j = 1; j <= jmax; j+=4) {

                // (flag[i][j] & C_F)
                mask = _mm_setr_ps((flag[i][j] & C_F),
                                   (flag[i][j+1] & C_F),
                                   (flag[i][j+2] & C_F),
                                   (flag[i][j+3] & C_F));

                //fixes issue where 1.0 represented true, now -nan does
                mask = _mm_cmpeq_ps(mask, _mm_set1_ps(16.0));

                // loads p[i][j], p[i][j+1], p[i][j+2], p[i][j+3],
                ap = _mm_loadu_ps(p[i] + j);
                // loads p[i+1][j], p[i+1][j+1], p[i+1][j+2], p[i+1][j+3],
                bp = _mm_loadu_ps(p[i+1] + j);
                // loads p[i-1][j], p[i-1][j+1], p[i-1][j+2], p[i-1][j+3],
                cp = _mm_loadu_ps(p[i-1] + j);
                // loads p[i][j+1], p[i][j+2], p[i][j+3], p[i][j+4],
                dp = _mm_loadu_ps(p[i] + j + 1);
                // loads p[i][j-1], p[i][j], p[i][j+1], p[i][j+2],
                ep = _mm_loadu_ps(p[i] + j - 1);
                // loads rhs[i][j], rhs[i][j+1], rhs[i][j+2], rhs[i][j+3]
                vec_rhs = _mm_loadu_ps(rhs[i] + j);
                // loads eps_N
                vec_eps_N = _mm_div_ps(_mm_setr_ps((float) (flag[i][j+1] & C_F), (float) (flag[i][j+2] & C_F), (float) (flag[i][j+3] & C_F), (float) (flag[i][j+4] & C_F)), _mm_set1_ps(16.0));
                // loads eps_S
                vec_eps_S = _mm_div_ps(_mm_setr_ps((float) (flag[i][j-1] & C_F), (float) (flag[i][j] & C_F), (float) (flag[i][j+1] & C_F), (float) (flag[i][j+2] & C_F)), _mm_set1_ps(16.0));
                // loads eps_E
                vec_eps_E = _mm_div_ps(_mm_setr_ps((float) (flag[i+1][j] & C_F), (float) (flag[i+1][j+1] & C_F), (float) (flag[i+1][j+2] & C_F), (float) (flag[i+1][j+3] & C_F)), _mm_set1_ps(16.0));
                // loads eps_W
                vec_eps_W = _mm_div_ps(_mm_setr_ps((float) (flag[i-1][j] & C_F), (float) (flag[i-1][j+1] & C_F), (float) (flag[i-1][j+2] & C_F), (float) (flag[i-1][j+3] & C_F)), _mm_set1_ps(16.0));

                // p[i][j]*p[i][j]
                vec_p0 = _mm_mul_ps(ap, ap);
                //find which ones are valid for adding to p0 sum
                result = _mm_or_ps(_mm_and_ps(mask, vec_p0), _mm_andnot_ps(mask, _mm_set1_ps(0.0)));
                //running hadd twice makes every element the sum of everything in vector
                sum = _mm_hadd_ps(result, result);
                sum = _mm_hadd_ps(sum, sum);
                //get sum from vector
                _mm_storeu_ps(temp_store, sum);
                p0 += temp_store[0];


                // (p[i+1][j]-p[i][j])
                one = _mm_sub_ps(bp, ap);
                // eps_E*(p[i+1][j]-p[i][j])
                two = _mm_mul_ps(vec_eps_E, one);
                // (p[i][j]-p[i-1][j])
                three = _mm_sub_ps(ap, cp);
                // eps_W*(p[i][j]-p[i-1][j])
                four = _mm_mul_ps(vec_eps_W, three);
                // (eps_E*(p[i+1][j]-p[i][j]) - eps_W*(p[i][j]-p[i-1][j]))
                five = _mm_sub_ps(two, four);
                // (eps_E*(p[i+1][j]-p[i][j]) - eps_W*(p[i][j]-p[i-1][j])) * rdx2
                six = _mm_mul_ps(five, vec_rdx2);
                // (p[i][j+1]-p[i][j])
                seven = _mm_sub_ps(dp, ap);
                // eps_N*(p[i][j+1]-p[i][j])
                eight = _mm_mul_ps(vec_eps_N, seven);
                // (p[i][j]-p[i][j-1])
                nine = _mm_sub_ps(ap, ep);
                // eps_S*(p[i][j]-p[i][j-1])
                ten = _mm_mul_ps(vec_eps_S, nine);
                // (eps_N*(p[i][j+1]-p[i][j]) - eps_S*(p[i][j]-p[i][j-1]))
                eleven = _mm_sub_ps(eight, ten);
                // (eps_N*(p[i][j+1]-p[i][j]) - eps_S*(p[i][j]-p[i][j-1])) * rdy2
                twelve = _mm_mul_ps(eleven, vec_rdy2);
                // (eps_E*(p[i+1][j]-p[i][j]) - eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                // (eps_N*(p[i][j+1]-p[i][j]) - eps_S*(p[i][j]-p[i][j-1])) * rdy2
                thirteen = _mm_add_ps(six, twelve);
                // add
                vec_add = _mm_sub_ps(thirteen, vec_rhs);
                // holds temp_res from the 4 array elements
                vec_temp_res = _mm_mul_ps(vec_add, vec_add);
                //find which ones are valid for adding to temp_res sum
                result = _mm_or_ps(_mm_and_ps(mask, vec_temp_res), _mm_andnot_ps(mask, _mm_set1_ps(0.0)));
                _mm_storeu_ps(temp_store, result);
                //sum up the values in vector
                sum = _mm_hadd_ps(result, result);
                sum = _mm_hadd_ps(sum, sum);
                //get sum from vector
                _mm_storeu_ps(temp_store, sum);
                temp_res += temp_store[0];
            }
            // //catch the rest
            // for (; j<=jmax; j++) {
            //     if (flag[i][j] & C_F) {
            //         /* moved here from fusing computing sum of squares */
            //         p0 += p[i][j]*p[i][j];
            //         /* only fluid cells */
            //         add = (eps_E*(p[i+1][j]-p[i][j]) -
            //             eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
            //             (eps_N*(p[i][j+1]-p[i][j]) -
            //             eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
            //         temp_res += add*add;
            //     }
            // }
        }
        /* more from dispursing sum of squares loop */
        p0 = sqrt(p0/ifull);
        if (p0 < 0.0001) { p0 = 1.0; }
        *res = sqrt((temp_res)/ifull)/p0;

        /* convergence? */
        if (*res<eps) break;
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

    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }
        }
    }
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
            }
        }
    }
}


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
        v[i][jmax] = 0.0;
        u[i][jmax+1] = u[i][jmax];

        v[i][0] = 0.0;
        u[i][0] = u[i][1];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & B_NSEW) {
                switch (flag[i][j]) {
                    case B_N:
                        v[i][j]   = 0.0;
                        u[i][j]   = -u[i][j+1];
                        u[i-1][j] = -u[i-1][j+1];
                        break;
                    case B_E:
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        v[i][j-1] = -v[i+1][j-1];
                        break;
                    case B_S:
                        v[i][j-1] = 0.0;
                        u[i][j]   = -u[i][j-1];
                        u[i-1][j] = -u[i-1][j-1];
                        break;
                    case B_W:
                        u[i-1][j] = 0.0;
                        v[i][j]   = -v[i-1][j];
                        v[i][j-1] = -v[i-1][j-1];
                        break;
                    case B_NE:
                        v[i][j]   = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j-1] = -v[i+1][j-1];
                        u[i-1][j] = -u[i-1][j+1];
                        break;
                    case B_SE:
                        v[i][j-1] = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        u[i-1][j] = -u[i-1][j-1];
                        break;
                    case B_SW:
                        v[i][j-1] = 0.0;
                        u[i-1][j] = 0.0;
                        v[i][j]   = -v[i-1][j];
                        u[i][j]   = -u[i][j-1];
                        break;
                    case B_NW:
                        v[i][j]   = 0.0;
                        u[i-1][j] = 0.0;
                        v[i][j-1] = -v[i-1][j-1];
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
