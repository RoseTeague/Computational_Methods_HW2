!Rosemary Teague
!00828351

!Module for solving n-d optimization problems with Newton's method and 2-d problems
!with bracket descent. Necessary cost function details are provided in separate cost
!module.
module hw2
  use cost
  implicit none
  integer :: itermax = 1000 !maximum number of iterations used by an optimizer
  real(kind=8) :: tol=1.0e-6 !stopping criteria for optimizer
  real(kind=8), allocatable :: jpath(:), xpath(:,:) !allocatable 1-d and 2-d arrays, should contain cost and location values used during optimization iterations

  contains


  subroutine newton(xguess,xf,jf)
    !Use Newton's method to minimize cost function, costj
    !input: xguess -- initial guess for loaction of minimum
    !output: xf -- computed location of minimum, jf -- computed minimum
    !Should also set module variables xpath and jpath appropriately
    implicit none
    integer :: i2,i1,N,NRHS,LDA,LDB,INFO
    real(kind=8), dimension(:,:), intent(in) :: xguess !do not need to explicitly specify dimension of input variable when subroutine is within a module
    real(kind=8), intent(out) :: xf(size(xguess),1),jf !location of minimum, minimum cost
    real(kind=8), allocatable :: Htemp(:,:),Gtemp(:,:),h(:,:),jg(:,:)
    real(kind=8) :: j1,j2,jh(size(xguess),size(xguess))
    logical :: flag_converged
    integer, allocatable, dimension(:) :: IPIV


    allocate(xpath(size(xguess),itermax),jpath(itermax))
    xpath(:,1)=xguess(:,1)
    N=size(xguess)
    NRHS = 1
    LDA = N
    LDB = N

  !  allocate(xpath(size(xguess):1),jpath(size(xguess)))
    allocate(Htemp(size(jh,1),size(jh,2)),IPIV(N),jg(N,1),Gtemp(N,1))
    allocate(h(N,1))
    !print *, x
    call costj(xpath(:,i1),j2)
    do i1=1,100!(itermax)
        j1=j2
        call costj_grad2d(xpath(:,i1),jg(:,1))
        call costj_hess2d(xpath(:,i1),jh)

        Htemp = -jh
        Gtemp = jg
        call dgesv(N, NRHS, Htemp, LDA, IPIV, Gtemp, LDB, INFO)
        !extract soln from Gtemp
        h = Gtemp(1:N,:)
        xpath(:,i1+1)=xpath(:,i1)+h(:,1)
        call costj(xpath(:,i1+1),j2)
        i2=i1
        call convergence_check(j1,j2,flag_converged)
        if (flag_converged) exit

    end do

    xf(:,1)=xpath(:,i2+1)
    jf=j2

  end subroutine newton


  subroutine bracket_descent(xguess,xf,jf)
    !Use bracket descent method to minimize cost function, costj
    !input: xguess -- initial guess for location of minimum
    !output: xf -- computed location of minimum, jf -- computed minimum
    !Should also set module variables xpath and jpath appropriately
    !Assumes size(xguess) = 2
    implicit none
    real(kind=8), dimension(2), intent(in) :: xguess
    real(kind=8), intent(out) :: xf(2),jf !location of minimum, minimum cost


  end subroutine bracket_descent



  subroutine bd_initialize(xguess,x3,j3)
    !given xguess, generates vertices (x3) and corresponding costs (j3) for initial
    !bracket descent step
    implicit none
    real(kind=8), intent(in) :: xguess(2)
    real(kind=8), intent(out) :: j3(3),x3(3,2) !location of minimum
    integer :: i1
    real(kind=8), parameter :: l=1.d0

    x3(1,1) = xguess(1)
    x3(2,1) = xguess(1)+l*sqrt(3.d0)/2
    x3(3,1) = xguess(1)-l*sqrt(3.d0)/2
    x3(1,2) = xguess(2)+l
    x3(2,2) = xguess(2)-l/2
    x3(3,2) = xguess(2)-l/2

    do i1=1,3
      call costj(x3(i1,:),j3(i1))
    end do
  end subroutine bd_initialize


  subroutine convergence_check(j1,j2,flag_converged)
    !check if costs j1 and j2 satisfy convergence criteria
    implicit none
    real(kind=8), intent(in) :: j1,j2
    real(kind=8) :: test
    logical, intent(out) :: flag_converged

    test = abs(j1-j2)/max(abs(j1),abs(j2),1.d0)
    if (test .le. tol) then
      flag_converged = .True.
    else
      flag_converged = .False.
    end if
  end subroutine convergence_check


end module hw2




program test
  use hw2
  implicit none
  real(kind=8), dimension(2,1) :: xguess, xf
  real(kind=8) :: jf!jh(size(xguess),size(xguess))

  xguess(:,1)=(/2.d0,10.d0/)

  call newton(xguess,xf,jf)
  !call costj_hess2d(xguess,jh)
  print *, 'test','x=',xf,'j=',jf

end program test
