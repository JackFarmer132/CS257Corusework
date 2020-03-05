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
    float temp[4];
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
    __m128 vec_f;
    __m128 vec_g;
    __m128 vec_du2dx;
    __m128 vec_duvdy;
    __m128 vec_laplu;

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
    // #pragma omp parallel for private(j) reduction(+:duvdx, dv2dy, laplv)
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {

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
              //loads f
              vec_f = _mm_loadu_ps(f[i] + j);

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



                // laplu/Re
                one = _mm_div_ps(vec_laplu, vec_Re);
                // (laplu/Re-du2dx-duvdy)
                two = _mm_sub_ps(one, _mm_add_ps(vec_du2dx, vec_duvdy));
                // del_t*(laplu/Re-du2dx-duvdy)
                three = _mm_mul_ps(vec_del_t, two);
                // u[i][j]+del_t*(laplu/Re-du2dx-duvdy)
                vec_f = _mm_add_ps(au, three);


            /* catches and sets all other types of cell */
            } else {
                f[i][j] = u[i][j];
            }
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {

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
                //loads f
                vec_g = _mm_loadu_ps(g[i] + j);


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
                // duvdy
                __m128 vec_duvdy = _mm_div_ps(eleven, twelve);
                _mm_store_ps(temp, vec_duvdy);

                






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
            /* catches and sets all other types of cell */
            } else {
                g[i][j] = v[i][j];
            }
        }
    }
}



/* Red/Black SOR to solve the poisson equation */
int poissonSolver(float **f, float **g, float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float del_t, float eps, int itermax, float omega,
    float *res, int ifull, int temp)
{
    int i, j, iter;
    float add, beta_mod;
    float p0 = 0.0;
    int rb; /* Red-black value. */

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    /* Red/Black SOR-iteration */
    for (iter = 0; iter < itermax; iter++) {
        for (rb = 0; rb <= 1; rb++) {
            #pragma omp parallel for private(j)
            for (i = 1; i <= imax; i++) {
                #pragma omp simd
                for (j = 1; j <= jmax; j++) {
                    if ((i+j) % 2 != rb) { continue; }
                    if (flag[i][j] & C_F) {
                      /* only compute if first iteration */
                      if (iter == 0){
                          /* moved in from computeRhs */
                          rhs[i][j] = (
                                       (f[i][j]-f[i-1][j])/delx +
                                       (g[i][j]-g[i][j-1])/dely
                                      ) / del_t;
                        }
                        /* modified star near boundary */
                        beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                        p[i][j] = (1.-omega)*p[i][j] -
                            beta_mod*(
                                  (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2
                                + (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                                - rhs[i][j]
                            );
                    }
                } /* end of j */
            } /* end of i */
          /* end of parallel section */
        } /* end of rb */
        //create temporary non-address based var to hold residual
        float temp_res = 0.0;
        #pragma omp parallel for private(j) reduction(+:temp_res, p0)
        for (i = 1; i <= imax; i++) {
            #pragma omp simd reduction(+:temp_res, p0)
            for (j = 1; j <= jmax; j++) {
                if (flag[i][j] & C_F) {
                    /* moved here from fusing computing sum of squares */
                    p0 += p[i][j]*p[i][j];
                    /* only fluid cells */
                    add = (eps_E*(p[i+1][j]-p[i][j]) -
                        eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                        (eps_N*(p[i][j+1]-p[i][j]) -
                        eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
                    temp_res += add*add;
                }
            }
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
    #pragma omp parallel for simd
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((i != imax) && (flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }
            /* only if both adjacent cells are fluid cells */
            if ((j != jmax) && (flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
            }
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
