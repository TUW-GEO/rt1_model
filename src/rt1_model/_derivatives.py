
# TODO clear caches !
self._d_surface_dummy_lambda.cache_clear()
self._d_surface_dummy_lambda.cache_clear()



    def surface_slope(self, dB=False, sig0=False):
        """
        Calculate the slope (dI_s/dt_0) of the (!monostatic!) surface-contribution.

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic slope of the
            surface-contribution
        """
        # evaluate the slope of the used brdf
        brdf_slope = self.SRF.brdf_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=1,
        )
        # evaluate the used brdf
        brdf_val = self.SRF.brdf(
            self.t_0,
            self.t_ex,
            self.p_0,
            self.p_ex,
            param_dict=self.param_dict,
        )

        # vegetated soil contribution
        I_vegs_slope = (
            self.I0
            * np.exp(-(2 * self.tau / self._mu_0))
            * (
                self._mu_0 * brdf_slope
                - (2 * self.tau / self._mu_0 + 1) * np.sin(self.t_0) * brdf_val
            )
        )

        # bare soil contribution
        I_bs_slope = self.I0 * (self._mu_0 * brdf_slope - np.sin(self.t_0) * brdf_val)

        I_slope = self.NormBRDF * (
            (1.0 - self.bsf) * I_vegs_slope + self.bsf * I_bs_slope
        )

        if sig0 is False and dB is False:
            return I_slope
        else:
            I_val = self.surface()
            if sig0 is True and dB is False:
                return 4.0 * np.pi * (self._mu_0 * I_slope - np.sin(self.t_0) * I_val)
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * I_slope / I_val
            elif sig0 is True and dB is True:
                return 10.0 / np.log(10) * (I_slope / I_val - np.tan(self.t_0))

    def surface_curv(self, dB=False, sig0=False):
        """
        Calculate curvature (d^2I_s/dt_0^2) of the (monostatic) surface-contribution.

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic curvature of the
            surface-contribution

        """
        # evaluate the slope of the used brdf
        brdf_curv = self.SRF.brdf_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=2,
        )
        # evaluate the slope of the used brdf
        brdf_slope = self.SRF.brdf_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=1,
        )
        # evaluate the used brdf
        brdf_val = self.SRF.brdf(
            self.t_0,
            self.t_ex,
            self.p_0,
            self.p_ex,
            param_dict=self.param_dict,
        )

        # vegetated soil contribution
        I_vegs_curv = (
            self.I0
            * np.exp(-(2.0 * self.tau / self._mu_0))
            * (
                self._mu_0 * brdf_curv
                - 2.0
                * np.sin(self.t_0)
                * brdf_slope
                * (2.0 * self.tau / self._mu_0 + 1.0)
                + (
                    4.0 * self.tau**2 / self._mu_0**3 * np.sin(self.t_0) ** 2
                    - 2.0 * self.tau
                    - self._mu_0
                )
                * brdf_val
            )
        )

        # bare soil contribution
        I_bs_curv = self.I0 * (
            self._mu_0 * brdf_curv
            - 2.0 * np.sin(self.t_0) * brdf_slope
            - self._mu_0 * brdf_val
        )

        I_curv = self.NormBRDF * ((1.0 - self.bsf) * I_vegs_curv + self.bsf * I_bs_curv)

        if sig0 is False and dB is False:
            return I_curv
        else:
            I_slope = self.surface_slope(dB=False, sig0=False)
            I_val = self.surface()
            if sig0 is True and dB is False:
                return (
                    4.0
                    * np.pi
                    * (
                        self._mu_0 * I_curv
                        - 2.0 * np.sin(self.t_0) * I_slope
                        - self._mu_0 * I_val
                    )
                )
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * (I_curv / I_val - I_slope**2 / I_val**2)
            elif sig0 is True and dB is True:
                return (
                    10.0
                    / np.log(10)
                    * (I_curv / I_val - I_slope**2 / I_val**2 - self._mu_0 ** (-2))
                )


    def volume_slope(self, dB=False, sig0=False):
        """
        Calculate the slope (dI_v/dt_0) of the (monostatic) volume-contribution.

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic slope of the
            volume-contribution

        """
        # evaluate the slope of the used phase-function
        p_slope = self.V.p_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=1,
        )

        # evaluate the used phase function
        p_val = self.V.p(
            self.t_0,
            self.t_ex,
            self.p_0,
            self.p_ex,
            param_dict=self.param_dict,
        )

        # volume contribution
        I_slope = (
            (1.0 - self.bsf)
            * self.I0
            * self.omega
            / 2.0
            * (
                (
                    np.exp(-(2 * self.tau / self._mu_0))
                    * 2
                    * self.tau
                    * np.sin(self.t_0)
                    / self._mu_0**2
                )
                * p_val
                + (1.0 - np.exp(-(2 * self.tau / self._mu_0))) * p_slope
            )
        )

        if sig0 is False and dB is False:
            return I_slope
        else:
            I_val = self.volume()
            if sig0 is True and dB is False:
                return 4.0 * np.pi * (self._mu_0 * I_slope - np.sin(self.t_0) * I_val)
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * I_slope / I_val
            elif sig0 is True and dB is True:
                return 10.0 / np.log(10) * (I_slope / I_val - np.tan(self.t_0))

    def volume_curv(self, dB=False, sig0=False):
        """
        Calculate curvature (d^2I_s/dt_0^2) of the (monostatic) volume-contribution.

        Parameters
        ----------
        dB : bool (default = False)
             Indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               Indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        array_like(float)
            Numerical value of the monostatic curvature of the
            volume-contribution
        """
        # evaluate the slope of the used brdf
        p_curv = self.V.p_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=2,
        )
        # evaluate the slope of the used brdf
        p_slope = self.V.p_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=1,
        )
        # evaluate the used brdf
        p_val = self.V.p(
            self.t_0,
            self.t_ex,
            self.p_0,
            self.p_ex,
            param_dict=self.param_dict,
        )

        I_curv = (
            (1.0 - self.bsf)
            * self.I0
            * self.omega
            / 2.0
            * (
                np.exp(-(2 * self.tau / self._mu_0))
                * (2 * self.tau / self._mu_0**3)
                * (
                    np.sin(self.t_0) ** 2
                    + 1.0
                    - 2.0 * self.tau / self._mu_0 * np.sin(self.t_0) ** 2
                )
                * p_val
                + (
                    np.exp(-(2 * self.tau / self._mu_0))
                    * 4.0
                    * self.tau
                    / self._mu_0**2
                    * np.sin(self.t_0)
                )
                * p_slope
                + (1 - np.exp(-(2 * self.tau / self._mu_0))) * p_curv
            )
        )

        if sig0 is False and dB is False:
            return I_curv
        else:
            I_slope = self.volume_slope(dB=False, sig0=False)
            I_val = self.volume()
            if sig0 is True and dB is False:
                return (
                    4.0
                    * np.pi
                    * (
                        self._mu_0 * I_curv
                        - 2.0 * np.sin(self.t_0) * I_slope
                        - self._mu_0 * I_val
                    )
                )
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * (I_curv / I_val - I_slope**2 / I_val**2)
            elif sig0 is True and dB is True:
                return (
                    10.0
                    / np.log(10)
                    * (I_curv / I_val - I_slope**2 / I_val**2 - self._mu_0 ** (-2))
                )

    def tot_slope(self, sig0=False, dB=False):
        """
        Calculate the (monostatic) slope of total contribution (surface + volume).

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        array_like(float)
            Numerical value of the monostatic slope of the
            total-contribution

        """
        I_slope = self.volume_slope(dB=False, sig0=False) + self.surface_slope(
            dB=False, sig0=False
        )

        if sig0 is False and dB is False:
            return I_slope
        else:
            I_val = self.volume() + self.surface()
            if sig0 is True and dB is False:
                return 4.0 * np.pi * (self._mu_0 * I_slope - np.sin(self.t_0) * I_val)
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * I_slope / I_val
            elif sig0 is True and dB is True:
                return 10.0 / np.log(10) * (I_slope / I_val - np.tan(self.t_0))

    def tot_curv(self, sig0=False, dB=False):
        """
        Calculate the (monostatic) curvature of total contribution (surface + volume).

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic curvature of the
            total-contribution

        """
        I_curv = self.volume_curv(dB=False, sig0=False) + self.surface_curv(
            dB=False, sig0=False
        )

        if sig0 is False and dB is False:
            return I_curv
        else:
            I_slope = self.volume_slope(dB=False, sig0=False) + self.surface_slope(
                dB=False, sig0=False
            )
            I_val = self.volume() + self.surface()
            if sig0 is True and dB is False:
                return (
                    4.0
                    * np.pi
                    * (
                        self._mu_0 * I_curv
                        - 2.0 * np.sin(self.t_0) * I_slope
                        - self._mu_0 * I_val
                    )
                )
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * (I_curv / I_val - I_slope**2 / I_val**2)
            elif sig0 is True and dB is True:
                return (
                    10.0
                    / np.log(10)
                    * (I_curv / I_val - I_slope**2 / I_val**2 - self._mu_0 ** (-2))
                )


    def _dvolume_dtau(self):
        """
        Get the derivative of the volume-contribution with respect to tau.

        Returns
        -------
        dvdt : array_like(float)
               Numerical value of dIvol/dtau for the given set of parameters

        """
        dvdt = (
            self.I0
            * self.omega
            * (self._mu_0 / (self._mu_0 + self._mu_ex))
            * (
                (1.0 / self._mu_0 + 1.0 / self._mu_ex ** (-1))
                * np.exp(-self.tau / self._mu_0 - self.tau / self._mu_ex)
            )
            * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        return (1.0 - self.bsf) * dvdt

    def _dvolume_domega(self):
        """
        Get the derivative of the volume-contribution with respect to omega.

        Returns
        -------
        dvdo : array_like(float)
               Numerical value of dIvol/domega for the given set of parameters

        """
        dvdo = (
            (self.I0 * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        return (1.0 - self.bsf) * dvdo

    def _dvolume_dbsf(self):
        """
        Get the derivative of the volume-contribution with respect to bsf.

        Returns
        -------
        dvdo : array_like(float)
               Numerical value of dIvol/dbsf for the given set of parameters

        """
        vol = (
            (self.I0 * self.omega * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * self.V.p(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        return -vol

    def _dvolume_dR(self):
        """
        Get the derivative of the volume-contribution with respect to NormBRDF.

        Returns
        -------
        dvdr : array_like(float)
               Numerical value of dIvol/dNormBRDF for the given set of parameters

        """
        dvdr = 0.0

        return dvdr

    def _dsurface_dtau(self):
        """
        Get the derivative of the surface-contribution with respect to tau.

        Returns
        -------
        dsdt : array_like(float)
               Numerical value of dIsurf/dtau for the given set of parameters

        """
        dsdt = (
            self.I0
            * (-1.0 / self._mu_0 - 1.0 / self._mu_ex)
            * np.exp(-self.tau / self._mu_0 - self.tau / self._mu_ex)
            * self._mu_0
            * self.SRF.brdf(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        # Incorporate BRDF-normalization factor
        dsdt = self.NormBRDF * (1.0 - self.bsf) * dsdt

        return dsdt

    def _dsurface_domega(self):
        """
        Get the derivative of the surface-contribution with respect to omega.

        Returns
        -------
        dsdo : array_like(float)
               Numerical value of dIsurf/domega for the given set of parameters

        """
        dsdo = 0.0

        return dsdo

    def _dsurface_dR(self):
        """
        Get the derivative of the surface-contribution with respect to NormBRDF.

        Returns
        -------
        dsdr : array_like(float)
               Numerical value of dIsurf/dNormBRDF for the given set of parameters

        """
        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.brdf(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        Isurf = (np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex))) * I_bs

        return (1.0 - self.bsf) * Isurf + self.bsf * I_bs

    def _dsurface_dbsf(self):
        """
        Numerical evaluation of the surface-contribution.

        (http://rt1.readthedocs.io/en/latest/theory.html#surface_contribution)

        Returns
        -------
        array_like(float)
            Numerical value of the surface-contribution for the
            given set of parameters

        """
        # bare soil contribution
        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.brdf(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        Isurf = (
            (np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * I_bs
            * np.ones_like(self.t_0)
        )

        return self.NormBRDF * (I_bs - Isurf)

    @lru_cache(20)
    def _d_surface_dummy_lambda(self, key):
        """
        Get a function to compute direct surface-contribution parameter derivatives.

        A cached lambda-function for computing the derivative of the surface-function
        with respect to a given parameter.

        Parameters
        ----------
        key : str
            the parameter to use.

        Returns
        -------
        callable
            A function that calculates the derivative with respect to key.

        """
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(self.param_dict.keys())

        return sp.lambdify(
            args,
            sp.diff(self.SRF._func, sp.Symbol(key)),
            modules=["numpy", "sympy"],
        )

    @lru_cache(20)
    def _d_volume_dummy_lambda(self, key):
        """
        Get a function to compute direct volume-contribution parameter derivatives.

        A cached lambda-function for computing the derivative of the volume-function
        with respect to a given parameter.

        Parameters
        ----------
        key : str
            the parameter to use.

        Returns
        -------
        callable
            A function that calculates the derivative with respect to key.

        """
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(self.param_dict.keys())

        return sp.lambdify(
            args,
            sp.diff(self.V._func, sp.Symbol(key)),
            modules=["numpy", "sympy"],
        )

    def _d_surface_ddummy(self, key):
        """
        Surface contribution derivative with respect to a given parameter (incl. bsf).

        Parameters
        ----------
        key : The parameter to use

        Returns
        -------
        array_like(float)
            Numerical value of dIsurf/dkey for the given set of parameters

        """
        dI_bs = (
            self.I0
            * self._mu_0
            * self._d_surface_dummy_lambda(key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
        )

        dI_s = (np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex))) * dI_bs

        return self.NormBRDF * ((1.0 - self.bsf) * dI_s + self.bsf * dI_bs)

    def _d_volume_ddummy(self, key):
        """
        Volume contribution derivative with respect to a given parameter (incl. bsf).

        Parameters
        ----------
        key : The parameter to use

        Returns
        -------
        array_like(float)
            Numerical value of dIvol/dkey for the given set of parameters

        """
        dIvol = (
            self.I0
            * self.omega
            * self._mu_0
            / (self._mu_0 + self._mu_ex)
            * (1.0 - np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * self._d_volume_dummy_lambda(key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
        )
        return (1.0 - self.bsf) * dIvol

    def jacobian(self, dB=False, sig0=False, param_list=["omega", "tau", "NormBRDF"]):
        """
        Return the jacobian of the total backscatter.

        (With respect to the parameters provided in param_list.)
        (default: param_list = ['omega', 'tau', 'NormBRDF'])

        The jacobian can be evaluated for measurements in linear or dB
        units, and for either intensity- or sigma_0 values.

        Note:
            The contribution of the interaction-term is currently
            not considered in the calculation of the jacobian!

        Parameters
        ----------
        dB : boolean (default = False)
             Indicator whether linear or dB units are used.
             The applied relation is given by:

             dI_dB(x)/dx =
             10 / [log(10) * I_linear(x)] * dI_linear(x)/dx

        sig0 : boolean (default = False)
               Indicator wheather intensity- or sigma_0-values are used
               The applied relation is given by:

               sig_0 = 4 * pi * cos(inc) * I

               where inc denotes the incident zenith-angle and I is the
               corresponding intensity
        param_list : list
                     a list of strings that correspond to the parameters
                     for which the jacobian should be evaluated.

                     possible values are: 'omega', 'tau' 'NormBRDF' and
                     any string corresponding to a sympy.Symbol used in the
                     definition of V or SRF

        Returns
        -------
        jac : array-like(float)
              The jacobian of the total backscatter with respect to
              omega, tau and NormBRDF

        """
        if sig0 is True and dB is False:
            norm = 4.0 * np.pi * np.cos(self.t_0)
        elif dB is True:
            norm = 10.0 / (np.log(10.0) * (self.surface() + self.volume()))
        else:
            norm = 1.0

        jac = []
        for key in param_list:
            if key == "omega":
                jac += [(self._dsurface_domega() + self._dvolume_domega()) * norm]
            elif key == "tau":
                jac += [(self._dsurface_dtau() + self._dvolume_dtau()) * norm]
            elif key == "NormBRDF":
                jac += [(self._dsurface_dR() + self._dvolume_dR()) * norm]
            elif key == "bsf":
                jac += [(self._dsurface_dbsf() + self._dvolume_dbsf()) * norm]
            elif key in self.param_dict:
                jac += [
                    (self._d_surface_ddummy(key) + self._d_volume_ddummy(key)) * norm
                ]
            else:
                assert False, (
                    "error in jacobian calculation... "
                    + str(key)
                    + " is not in param_dict"
                )

        return jac




