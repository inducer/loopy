! straight from gNUMA, do not modify in loopy

subroutine strongVolumeKernelR(elements, &
     volumeGeometricFactors, D, Q, gradQ, rhsQ)
  implicit none

  integer*4 elements
  integer*4 e,i,j,k,n
  datafloat volumeGeometricFactors(Nq, Nq, Nq, 11, elements)
  datafloat D(Nq,Nq)
  datafloat Q(Nq, Nq, Nq, 8, elements)
  datafloat rhsQ(Nq, Nq, Nq, 8, elements)
  datafloat gradQ(Nq, Nq, Nq, 8, 3, elements) ! FIXME CHECK, UNUSED FOR NOW

  datafloat U, V, W, R, T, Qa, Qw, P

  datafloat Uflux, Vflux, Wflux, Rflux, Tflux, Qaflux, Qwflux

  datafloat Jrx, Jry, Jrz, Jinv
  datafloat UdotGradR
  datafloat JinvD

  do e = 1, elements

    do k = 1, Nq
      do j = 1, Nq
        do i = 1, Nq
          do n = 1,Nq

!$loopy begin tagged: local_prep
            U  = Q(n, j, k, 1, e)
            V  = Q(n, j, k, 2, e)
            W  = Q(n, j, k, 3, e)
            R  = Q(n, j, k, 5, e)
            T  = Q(n, j, k, 6, e)
            Qa = Q(n, j, k, 7, e)
            Qw = Q(n, j, k, 8, e)

            Jrx  = volumeGeometricFactors(n, j, k, 1, e)
            Jry  = volumeGeometricFactors(n, j, k, 2, e)
            Jrz  = volumeGeometricFactors(n, j, k, 3, e)

            Jinv = volumeGeometricFactors(i, j, k, 10, e)

            P = p_p0*(p_R*T/p_p0) ** p_Gamma
            UdotGradR = (Jrx*U + Jry*V + Jrz*W)/R
!$loopy end tagged: local_prep

            JinvD = Jinv*D(i,n)

!$loopy begin tagged: compute_fluxes
            Uflux = U*UdotGradR + Jrx*P
            Vflux = V*UdotGradR + Jry*P
            Wflux = W*UdotGradR + Jrz*P
            Rflux = R*UdotGradR
            Tflux = T*UdotGradR

            Qaflux = Qa*UdotGradR
            Qwflux = Qw*UdotGradR
!$loopy end tagged: compute_fluxes

            rhsQ(i, j, k, 1, e) = rhsQ(i, j, k, 1, e) - JinvD*Uflux
            rhsQ(i, j, k, 2, e) = rhsQ(i, j, k, 2, e) - JinvD*Vflux
            rhsQ(i, j, k, 3, e) = rhsQ(i, j, k, 3, e) - JinvD*Wflux

            rhsQ(i, j, k, 5, e) = rhsQ(i, j, k, 5, e) - JinvD*Rflux
            rhsQ(i, j, k, 6, e) = rhsQ(i, j, k, 6, e) - JinvD*Tflux
            rhsQ(i, j, k, 7, e) = rhsQ(i, j, k, 7, e) - JinvD*Qaflux
            rhsQ(i, j, k, 8, e) = rhsQ(i, j, k, 8, e) - JinvD*Qwflux
          end do
        end do
      end do
    end do
  end do
end subroutine strongVolumeKernelR

subroutine strongVolumeKernelS(elements, &
     volumeGeometricFactors, D, Q, gradQ, rhsQ)
  implicit none

  integer*4 elements
  integer*4 e,i,j,k,n
  datafloat volumeGeometricFactors(Nq, Nq, Nq, 11, elements)
  datafloat D(Nq,Nq)
  datafloat Q(Nq, Nq, Nq, 8, elements)
  datafloat rhsQ(Nq, Nq, Nq, 8, elements)
  datafloat gradQ(Nq, Nq, Nq, 8, 3, elements) ! FIXME CHECK, UNUSED FOR NOW

  datafloat U, V, W, R, T, Qa, Qw, P

  datafloat Uflux, Vflux, Wflux, Rflux, Tflux, Qaflux, Qwflux

  datafloat Jsx, Jsy, Jsz, Jinv
  datafloat UdotGradS
  datafloat JinvD

  do e = 1, elements

    do k = 1, Nq
      do j = 1, Nq
        do i = 1, Nq
          do n = 1,Nq

!$loopy begin tagged: local_prep
            U  = Q(i, n, k, 1, e)
            V  = Q(i, n, k, 2, e)
            W  = Q(i, n, k, 3, e)
            R  = Q(i, n, k, 5, e)
            T  = Q(i, n, k, 6, e)
            Qa = Q(i, n, k, 7, e)
            Qw = Q(i, n, k, 8, e)

            Jsx  = volumeGeometricFactors(i, n, k, 4, e)
            Jsy  = volumeGeometricFactors(i, n, k, 5, e)
            Jsz  = volumeGeometricFactors(i, n, k, 6, e)

            Jinv = volumeGeometricFactors(i, j, k, 10, e)

            P = p_p0*(p_R*T/p_p0) ** p_Gamma
            UdotGradS = (Jsx*U + Jsy*V + Jsz*W)/R
!$loopy end tagged: local_prep

            JinvD = Jinv*D(j,n)

!$loopy begin tagged: compute_fluxes
            Uflux = U*UdotGradS + Jsx*P
            Vflux = V*UdotGradS + Jsy*P
            Wflux = W*UdotGradS + Jsz*P
            Rflux = R*UdotGradS
            Tflux = T*UdotGradS

            Qaflux = Qa*UdotGradS
            Qwflux = Qw*UdotGradS
!$loopy end tagged: compute_fluxes

            rhsQ(i, j, k, 1, e) = rhsQ(i, j, k, 1, e) - JinvD*Uflux
            rhsQ(i, j, k, 2, e) = rhsQ(i, j, k, 2, e) - JinvD*Vflux
            rhsQ(i, j, k, 3, e) = rhsQ(i, j, k, 3, e) - JinvD*Wflux

            rhsQ(i, j, k, 5, e) = rhsQ(i, j, k, 5, e) - JinvD*Rflux
            rhsQ(i, j, k, 6, e) = rhsQ(i, j, k, 6, e) - JinvD*Tflux
            rhsQ(i, j, k, 7, e) = rhsQ(i, j, k, 7, e) - JinvD*Qaflux
            rhsQ(i, j, k, 8, e) = rhsQ(i, j, k, 8, e) - JinvD*Qwflux
          end do
        end do
      end do
    end do
  end do
end subroutine strongVolumeKernelS

subroutine strongVolumeKernelT(elements, &
      volumeGeometricFactors, D, Q, gradQ, rhsQ)
  implicit none

  integer*4 elements
  datafloat volumeGeometricFactors(Nq, Nq, Nq, 11, elements)
  datafloat D(Nq,Nq)
  datafloat Q(Nq, Nq, Nq, 8, elements)
  datafloat rhsQ(Nq, Nq, Nq, 8, elements)
  datafloat gradQ(Nq, Nq, Nq, 8, 3, elements) ! FIXME CHECK, UNUSED FOR NOW

  datafloat U, V, W, R, T, Qa, Qw, P, UdotGradT
  datafloat Jtx, Jty, Jtz, Jinv, JinvD
  datafloat Uflux, Vflux, Wflux, Rflux, Tflux, Qaflux, Qwflux

  integer e, j, k, i, n

  do e = 1, elements
    do j = 1, Nq
      do k = 1, Nq
        do i = 1, Nq
          do n = 1,Nq
!$loopy begin tagged: local_prep
            U  = Q(i, j, n, 1, e)
            V  = Q(i, j, n, 2, e)
            W  = Q(i, j, n, 3, e)
            R  = Q(i, j, n, 5, e)
            T  = Q(i, j, n, 6, e)
            Qa = Q(i, j, n, 7, e)
            Qw = Q(i, j, n, 8, e)

            Jtx  = volumeGeometricFactors(i, j, n, 7, e)
            Jty  = volumeGeometricFactors(i, j, n, 8, e)
            Jtz  = volumeGeometricFactors(i, j, n, 9, e)

            Jinv = volumeGeometricFactors(i, j, k, 10, e)

            P = p_p0*(p_R*T/p_p0) ** p_Gamma
            UdotGradT = (Jtx*U + Jty*V + Jtz*W)/R
!$loopy end tagged: local_prep

            JinvD = Jinv*D(k,n)

!$loopy begin tagged: compute_fluxes
            Uflux = U*UdotGradT + Jtx*P
            Vflux = V*UdotGradT + Jty*P
            Wflux = W*UdotGradT + Jtz*P
            Rflux = R*UdotGradT
            Tflux = T*UdotGradT

            Qaflux = Qa*UdotGradT
            Qwflux = Qw*UdotGradT
!$loopy end tagged: compute_fluxes

            rhsQ(i, j, k, 1, e) = rhsQ(i, j, k, 1, e) - JinvD*Uflux
            rhsQ(i, j, k, 2, e) = rhsQ(i, j, k, 2, e) - JinvD*Vflux
            rhsQ(i, j, k, 3, e) = rhsQ(i, j, k, 3, e) - JinvD*Wflux

            rhsQ(i, j, k, 5, e) = rhsQ(i, j, k, 5, e) - JinvD*Rflux
            rhsQ(i, j, k, 6, e) = rhsQ(i, j, k, 6, e) - JinvD*Tflux
            rhsQ(i, j, k, 7, e) = rhsQ(i, j, k, 7, e) - JinvD*Qaflux
            rhsQ(i, j, k, 8, e) = rhsQ(i, j, k, 8, e) - JinvD*Qwflux
          end do
        end do
      end do
    end do
  end do
end subroutine strongVolumeKernelT

