program speed
  implicit none

  interface
    subroutine grad(n,u, du)
      integer, intent(in) :: n
      real, dimension(n,n), intent(in) :: u 
      real, dimension(2,n,n), intent(out) :: du 
    end subroutine
  end interface

end

subroutine grad(u, du, n)
  real, dimension(n,n), intent(in) :: u 
  real, dimension(2,n,n), intent(inout) :: du  
  integer, intent(in) :: n

  du(2,:,1:(n-1)) = u(:,2:n) - u(:,1:(n-1)) 
  du(1,1:(n-1),:) = u(2:n,:) - u(1:(n-1),:)
  !!$OMP DO
  !do i=1,n
  !  du(1,n,i) = 0.0
  !  du(2,n,i) = 0.0
  !  du(1,1:(n-1),i) = u(2:n,i) - u(1:(n-1),i)
  !  du(2,i,1:(n-1)) = u(i,2:n) - u(i,1:(n-1)) 
  !enddo
  !!$OMP END DO
end subroutine

subroutine div(du, u, n)
  real, dimension(2,n,n), intent(in) :: du
  real, dimension(n,n), intent(inout) :: u 
  integer, intent(in) :: n
  
  u(1,:) = du(1,1,:)
  u(n,:) = -du(1,n-1,:)

  u(2:(n-1),:) = du(1,2:(n-1),:) - du(1,1:(n-2),:)

  u(:,1) = u(:,1) + du(2,:,1)
  u(:,n) = u(:,n) - du(2,:,n-1)
  
  u(:, 2:(n-1)) = u(:, 2:(n-1)) + du(2,:,2:(n-1)) - du(2,:,1:(n-2))
end subroutine

subroutine norm1(du, u, n)
  real, dimension(2,n,n), intent(in) :: du
  real, dimension(n,n), intent(inout) :: u 
  integer, intent(in) :: n
  u = sqrt(du(1,:,:)**2 + du(2,:,:)**2)
end subroutine

subroutine quad_denoise(u, Au, w, lam, n)
  real, dimension(n,n), intent(in) :: u
  real, dimension(n,n), intent(inout) :: Au
  real, dimension(2,n,n), intent(inout) :: w
  real, intent(in) :: lam
  integer, intent(in) :: n
  
  call grad(u,w,n)
  call div(w,Au,n)

  Au = u - lam * Au
end subroutine

!subroutine quad_deconvolve(u, Au, w, fker, fker_star, lam, n)
!  real, dimension(n,n), intent(in) :: u
!  complex, dimension(n,n), intent(in) :: fker
!  complex, dimension(n,n), intent(in) :: fker_star
!  real, dimension(n,n), intent(inout) :: Au
!  real, dimension(2,n,n), intent(inout) :: w
!  real, intent(in) :: lam
!  integer, intent(in) :: n
!  
!  call grad(u,w,n)
!  call div(w,Au,n)
!  Au = u - lam * Au
!end subroutine

subroutine quadreg_denoise(f, u, lam, tol, n)
  real, dimension(n,n), intent(in) :: f
  real, dimension(n,n), intent(inout) :: u
  real, intent(in) :: lam
  real, intent(in) :: tol
  integer, intent(in) :: n
  
  real, dimension(n,n) :: p
  real, dimension(n,n) :: r
  real, dimension(n,n) :: Ap
  real, dimension(2,n,n) :: w ! Work vector needed in quad
  real :: alpha, beta, r0, rr_curr, rr_next
  integer :: nn, incx, incy, i, maxiter
  real :: snrm2, sdot
  external snrm2, saxpy, sdot
  nn = n*n
  !u = 0.0
  Ap = 0.0
  incx = 1
  incy = 1
  maxiter=100

  call quad_denoise(u, Ap, w, lam, n)
  r = f - Ap
  p = r
  rr_next = snrm2(nn, r, incx)**2
  r0 = rr_next
  do i=1,maxiter
    call quad_denoise(p, Ap, w, lam, n)
    rr_curr = rr_next
    alpha = rr_curr/sdot(nn,p,incx,Ap,incy)
    call saxpy(nn, alpha, p, incx, u, incy) !u = u + alpha * p
    call saxpy(nn, -alpha, Ap, incx, r, incy) !r = r - alpha * Ap
    rr_next = snrm2(nn,r,incx)**2
    if (rr_next/r0 < tol) then
      exit
    endif
    beta = rr_next/rr_curr
    p = r + beta*p
  enddo
  
end subroutine

!subroutine quadreg_deconvolve(f, res, ker, lam, tol, n, m)
!  real, dimension(n,n), intent(in) :: f
!  real, dimension(m,m), intent(in) :: ker
!  real, dimension(m/2 + 1,m), intent(inout) :: res
!  real, intent(in) :: lam
!  real, intent(in) :: tol
!  integer, intent(in) :: n
!  
!  real, dimension(n,n) :: p
!  real, dimension(n,n) :: r
!  complex, dimension(m/2 + 1,m) :: fker
!  integer*8 :: plan
!  real, dimension(n,n) :: Ap
!  real, dimension(2,n,n) :: w ! Work vector needed in quad
!  real :: alpha, beta, r0, rr_curr, rr_next
!  integer :: nn, incx, incy, i, maxiter
!  real :: snrm2, sdot
!  external snrm2, saxpy, sdot
!  !external sfftw_plan_dft_r2c_2d, sfftw_execute_dft_r2c, destroy_plan
!  nn = n*n
!  fker = 0.0
!  Ap = 0.0
!  incx = 1
!  incy = 1
!  maxiter=100
!  
!  call sfftw_plan_dft_r2c_2d(plan, m,m, ker, fker, FFTW_ESTIMATE)
!  call sfftw_execute_dft_r2c(plan, ker, fker)
!  call destroy_plan(plan)
!  res = fker
!  !call quad_denoise(u, Ap, w, lam, n)
!  !r = f - Ap
!  !p = r
!  !rr_next = snrm2(nn, r, incx)**2
!  !r0 = rr_next
!  !do i=1,maxiter
!  !  call quad_denoise(p, Ap, w, lam, n)
!  !  rr_curr = rr_next
!  !  alpha = rr_curr/sdot(nn,p,incx,Ap,incy)
!  !  call saxpy(nn, alpha, p, incx, u, incy) !u = u + alpha * p
!  !  call saxpy(nn, -alpha, Ap, incx, r, incy) !r = r - alpha * Ap
!  !  rr_next = snrm2(nn,r,incx)**2
!  !  if (rr_next/r0 < tol) then
!  !    exit
!  !  endif
!  !  beta = rr_next/rr_curr
!  !  p = r + beta*p
!  !enddo
!  !res = u
!  
!end subroutine

subroutine chambollepock_denoise(f, u, lam, tau, sig, theta, acc, tol, n)
  real, dimension(n,n), intent(in) :: f
  real, dimension(n,n), intent(inout) :: u
  real, intent(in) :: lam
  real, intent(inout) :: tau
  real, intent(inout) :: sig
  real, intent(inout) :: theta
  logical, intent(in) :: acc
  real, intent(in) :: tol
  integer, intent(in) :: n
  
  real, dimension(2,n,n) :: p
  real, dimension(n,n) :: divp
  real, dimension(n,n) :: u_hat
  real, dimension(2,n,n) :: gradu
  real :: gam
  integer :: nn, incx, incy, maxiter
  external saxpy, snrm2

  nn = n*n 

  incx = 1
  incy = 1

  u_hat = u
  gradu = 0
  p = 0
  divp = 0
  
  gam = 0.5
  maxiter = 100
  do i=1,maxiter
    call grad(u_hat, gradu, n)
    u_hat = u
    call saxpy(2*nn, sig, gradu, incx, p, incy)
    call norm1(p, divp, n)
    divp = max(lam, divp)
    p(1,:,:) = lam * p(1,:,:)/divp
    p(2,:,:) = lam * p(2,:,:)/divp
    call div(p, divp, n)
    u = 1.0/(1.0 + tau) * (u + tau * divp + tau * f)
    if (acc) then
      theta = 1.0/sqrt(1.0 + 2.0*gam*tau)
      tau = theta*tau
      sig = sig/theta
    endif
    u_hat = u - u_hat
    if(snrm2(nn,u_hat, incx)**2/nn < tol) then
      exit
    endif
    u_hat = u + theta*u_hat
  enddo
end subroutine
