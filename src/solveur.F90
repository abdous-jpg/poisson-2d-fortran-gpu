! Solveur de Jacobi pour -Laplacien(u) = f sur [0,1]^2, u = 0 au bord.
! Solution exacte : u(x,y) = sin(pi x) sin(pi y), donc f = 2 pi^2 sin(pi x) sin(pi y).
! Grille de n x n points (bords compris), pas h = 1/(n-1).
module solveur
  use precision
  implicit none
  real(wp), parameter :: pi = 3.14159265358979323846_wp
contains

  subroutine initialiser(n, u, f)
    integer, intent(in) :: n
    real(wp), intent(out) :: u(n,n), f(n,n)
    integer :: i, j
    real(wp) :: h, x, y
    h = 1.0_wp / real(n-1, wp)
    do j = 1, n
      y = real(j-1, wp) * h
      do i = 1, n            ! indice de gauche = boucle interne (ordre colonne)
        x = real(i-1, wp) * h
        f(i,j) = 2.0_wp * pi**2 * sin(pi*x) * sin(pi*y)
        u(i,j) = 0.0_wp
      end do
    end do
  end subroutine initialiser

  ! Fait au plus maxit iterations de Jacobi ; s'arrete si l'ecart max entre
  ! deux iterations passe sous tol (tol <= 0 : nombre d'iterations fixe).
  subroutine jacobi(n, u, f, tol, maxit, iters, ecart)
    integer, intent(in) :: n, maxit
    real(wp), intent(inout) :: u(n,n)
    real(wp), intent(in) :: f(n,n), tol
    integer, intent(out) :: iters
    real(wp), intent(out) :: ecart
    real(wp), allocatable :: v(:,:)
    real(wp) :: h2
    integer :: i, j, k

    allocate(v(n,n))
    v = u
    h2 = (1.0_wp / real(n-1, wp))**2
    iters = 0
    ecart = huge(1.0_wp)

#ifdef USE_GPU
    ! Les tableaux restent sur le GPU pendant toutes les iterations :
    ! sans cette region, chaque iteration recopierait u et f CPU <-> GPU.
    !$omp target data map(tofrom: u) map(to: f) map(alloc: v)
    !$omp target teams distribute parallel do collapse(2)
    do j = 1, n
      do i = 1, n
        v(i,j) = u(i,j)
      end do
    end do
#endif
    do k = 1, maxit
      ecart = 0.0_wp
#ifdef USE_GPU
      !$omp target teams distribute parallel do collapse(2) reduction(max:ecart)
#else
      !$omp parallel do collapse(2) reduction(max:ecart)
#endif
      do j = 2, n-1
        do i = 2, n-1
          v(i,j) = 0.25_wp * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h2 * f(i,j))
          ecart = max(ecart, abs(v(i,j) - u(i,j)))
        end do
      end do
#ifdef USE_GPU
      !$omp target teams distribute parallel do collapse(2)
#else
      !$omp parallel do collapse(2)
#endif
      do j = 2, n-1
        do i = 2, n-1
          u(i,j) = v(i,j)
        end do
      end do
      iters = k
      if (tol > 0.0_wp .and. ecart < tol) exit
    end do
#ifdef USE_GPU
    !$omp end target data
#endif
    deallocate(v)
  end subroutine jacobi

  ! Erreur L2 discrete (moyenne quadratique sur les noeuds) par rapport a la solution exacte
  function erreur_l2(n, u) result(err)
    integer, intent(in) :: n
    real(wp), intent(in) :: u(n,n)
    real(wp) :: err, h, x, y, s
    integer :: i, j
    h = 1.0_wp / real(n-1, wp)
    s = 0.0_wp
    do j = 1, n
      y = real(j-1, wp) * h
      do i = 1, n
        x = real(i-1, wp) * h
        s = s + (u(i,j) - sin(pi*x)*sin(pi*y))**2
      end do
    end do
    err = sqrt(s / real(n*n, wp))
  end function erreur_l2

end module solveur
