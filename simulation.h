void computeTentativeVelocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re);

void computeRhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely);

int poissonSolver(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull);

void updateVelocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely);

void setTimestepInterval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau);

void applyBoundaryConditions(float **u, float **v, char **flag,
    int imax, int jmax, float ui, float vi);
