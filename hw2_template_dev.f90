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


  subroutine newton(xguess,xf,jf,i2)
    !Use Newton's method to minimize cost function, costj
    !input: xguess -- initial guess for loaction of minimum
    !output: xf -- computed location of minimum, jf -- computed minimum
    !Should also set module variables xpath and jpath appropriately
    implicit none
    real(kind=8), dimension(:,:), intent(in) :: xguess !do not need to explicitly specify dimension of input variable when subroutine is within a module
    integer, intent(out) :: i2
    real(kind=8), intent(out) :: xf(size(xguess),1),jf !location of minimum, minimum cost
    real(kind=8), allocatable :: Htemp(:,:),Gtemp(:,:),h(:,:),jg(:,:),jh(:,:),xpath2(:,:)
    real(kind=8) :: j1,j2
    integer :: l,i1,N,NRHS,LDA,LDB,INFO
    logical :: flag_converged
    integer, allocatable, dimension(:) :: IPIV

    N=size(xguess)
    NRHS = 1
    LDA = N
    LDB = N

    if (allocated(jpath)) deallocate(jpath)
    if (allocated(xpath)) deallocate(xpath)

    allocate(xpath2(N,itermax),jpath(itermax))
    xpath2(:,1)=xguess(:,1)


    allocate(jh(N,N),Htemp(N,N),IPIV(N),jg(N,1),Gtemp(N,1),h(N,1))

    call costj(xpath2(:,i1),j2)
    do i1=1,(itermax)
        jpath(i1)=j2
        call costj_grad2d(xpath2(:,i1),jg(:,1))
        call costj_hess2d(xpath2(:,i1),jh)

        Htemp = -jh
        Gtemp = jg
        call dgesv(N, NRHS, Htemp, LDA, IPIV, Gtemp, LDB, INFO)
        !extract soln from Gtemp
        h = Gtemp(1:N,:)
        xpath2(:,i1+1)=xpath2(:,i1)+h(:,1)
        call costj(xpath2(:,i1+1),j2)
        l=i1
        call convergence_check(jpath(i1),j2,flag_converged)
        if (flag_converged) exit

    end do

    i2=l
    allocate(xpath(N,i2+1))
    xpath(:,:)=xpath2(:,1:i2+1)
    xf(:,1)=xpath(:,i2+1)
    jf=j2

    deallocate(xpath2,jh,Htemp,IPIV,jg,Gtemp,h)
  end subroutine newton


  subroutine bracket_descent(xguess,xf,jf,i2)
    !Use bracket descent method to minimize cost function, costj
    !input: xguess -- initial guess for location of minimum
    !output: xf -- computed location of minimum, jf -- computed minimum
    !Should also set module variables xpath and jpath appropriately
    !Assumes size(xguess) = 2
    implicit none
    real(kind=8), dimension(2), intent(in) :: xguess
    real(kind=8), intent(out) :: xf(2),jf !location of minimum, minimum cost
    integer, intent(out) :: i2
    real(kind=8), dimension(2) :: va, vb, vc, vatemp,vbtemp,vctemp,xm,xstar,xstar3,l
    real(kind=8) :: Ja, Jb, Jc,Jatemp,Jbtemp,Jctemp,Jstar,Jstar2,Jstar3,Jstar4,j1,j2
    real(kind=8) :: xpath2(2,itermax), jpath2(itermax)
    integer :: i1
    logical :: flag_converged

    va(1)=xguess(1)
    va(2)=xguess(2)+1.d0
    vb(1)=xguess(1)-sqrt(3.d0)/2.d0
    vb(2)=xguess(2)-1.d0/2.d0
    vc(1)=xguess(1)+sqrt(3.d0)/2.d0
    vc(2)=xguess(2)-1.d0/2.d0

    j2=100

    if (allocated(jpath)) deallocate(jpath)
    if (allocated(xpath)) deallocate(xpath)

    xpath2(:,1)=xguess
    call costj(xpath2(:,1),jpath2(1))


    call costj(va,Ja)
    call costj(vb,Jb)
    call costj(vc,Jc)
    j2=abs(Ja)+abs(Jb)+abs(Jc)

    do i1=1,itermax
      j1=j2

      if (Jc>Jb) then
        Jctemp=Jb; vctemp=vb
        Jbtemp=Jc; vbtemp=vc
      else
        Jctemp=Jc; vctemp=vc
        Jbtemp=Jb; vbtemp=vb
      end if

      if (Jbtemp>Ja) then
        Jatemp=Jbtemp; vatemp=vbtemp
        Jbtemp=Ja; vbtemp=va
        if (Jctemp>Jbtemp) then
          Ja=Jatemp; va = vatemp
          Jb=Jctemp; vb = vctemp
          Jc=Jbtemp; vc = vbtemp
        else
          Ja=Jatemp; va = vatemp
          Jb=Jbtemp; vb = vbtemp
          Jc=Jctemp; vc = vctemp
        end if
      else
        Jb = Jbtemp; vb = vbtemp
        Jc = Jctemp; vc = vctemp
      end if


      xm=(vb+vc)/2.d0
      l=xm-va
      xstar=va+2*l

      call costj(xstar,Jstar)

      if (Jstar .lt. Jc) then
        call costj(xstar+l,Jstar2)
        if (Jstar2 .lt. Jstar) then
          va = xstar+l
          Ja = Jstar2
        else
          va = xstar
          Ja = Jstar
        end if
      elseif (Jstar .le. Ja .and. Jstar .ge. Jc) then
        va = xstar
        Ja = Jstar
      elseif (Jstar .gt. Ja) then
        xstar3=xstar-l/2
        call costj(xstar3,Jstar3)
        if (Jstar3 .lt. Ja) then
          va = xstar3
          Ja = Jstar3
        else
          call costj(xstar-3*l/2,Jstar4)
          if (Jstar4 .lt. Ja) then
            va = xstar-3*l/2
            Ja = Jstar4
          else
            va = (va+vc)/2.d0
            vb = (vb+vc)/2.d0
          end if
        end if
      end if

      call costj(va,Ja)
      call costj(vb,Jb)
      call costj(vc,Jc)

      j2=abs(Ja)+abs(Jb)+abs(Jc)

      xpath2(:,i1+1)=(va+vb+vc)/3
      !xpath2(2,i1+1)=(va(2)+vb(2)+vc(2))/3
      call costj(xpath2(:,i1+1),jpath2(i1+1))

      i2=i1

    call convergence_check(j1,j2,flag_converged)
    if (flag_converged) exit

    end do

  allocate(xpath(2,i2+1),jpath(i2+1))
  xpath(:,:)=xpath2(:,1:i2+1)
  jpath(:)=jpath2(1:i2+1)
  xf=xpath(:,i2+1)
  jf=Jpath(size(jpath))

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
  use cost
  implicit none
  integer :: i2
  real(kind=8), dimension(2) :: xguess, xf
  real(kind=8) :: jf!jh(size(xguess),size(xguess))

  xguess(:)=(/-100.d0,-3.d0/)

  call bracket_descent(xguess,xf,jf,i2)
  !call newton(xguess,xf,jf,i2)
  !call costj_hess2d(xguess,jh)
  print *, 'test','x=',xf,'j=',jf
!  print *, 'jpath= ',jpath

end program test
