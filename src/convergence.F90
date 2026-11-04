! Etude de convergence : erreur L2 sur la solution exacte pour plusieurs maillages.
! Echoue (code de sortie 1) si l'ordre observe s'ecarte de 2.
program convergence
  use precision
  use solveur
  implicit none
  integer, parameter :: nb = 4
  integer :: tailles(nb) = [17, 33, 65, 129]
  real(wp), allocatable :: u(:,:), f(:,:)
  real(wp) :: err(nb), ecart, ordre
  integer :: m, n, iters, unite

  open(newunit=unite, file='results/convergence.csv', status='replace')
  write(unite,'(a)') 'n,h,iterations,erreur_l2,ordre'
  print '(a)', '    n         h   iterations    erreur L2   ordre'
  do m = 1, nb
    n = tailles(m)
    allocate(u(n,n), f(n,n))
    call initialiser(n, u, f)
    call jacobi(n, u, f, 1.0e-12_wp, 10000000, iters, ecart)
    err(m) = erreur_l2(n, u)
    if (m > 1) then
      ordre = log(err(m-1)/err(m)) / log(real(tailles(m)-1, wp)/real(tailles(m-1)-1, wp))
    else
      ordre = 0.0_wp
    end if
    print '(i5, es10.3, i13, es13.3, f8.2)', n, 1.0/real(n-1), iters, err(m), ordre
    write(unite,'(i0,",",es12.5,",",i0,",",es12.5,",",f6.3)') n, 1.0/real(n-1), iters, err(m), ordre
    deallocate(u, f)
  end do
  close(unite)
  if (abs(ordre - 2.0_wp) > 0.1_wp) then
    print '(a, f6.2)', 'ECHEC : ordre observe ', ordre
    error stop 1
  end if
  print '(a)', 'OK : ordre 2 retrouve'
end program convergence
