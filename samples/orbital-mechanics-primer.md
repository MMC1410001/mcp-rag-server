# Orbital Mechanics: A Short Primer

## 1. Kepler's Laws

**First law (ellipses).** Every planet moves in an ellipse with the Sun at one focus. The
eccentricity `e` runs from 0 (a circle) to just under 1 (a very elongated ellipse).

**Second law (equal areas).** A line joining a planet to the Sun sweeps out equal areas in
equal times. A body therefore moves fastest at periapsis and slowest at apoapsis.

**Third law (harmonies).** The square of the orbital period is proportional to the cube of
the semi-major axis: `T² ∝ a³`.

## 2. The Vis-Viva Equation

Orbital speed at any radius `r` follows from conservation of energy:

    v² = GM (2/r − 1/a)

where `GM` is the standard gravitational parameter and `a` the semi-major axis. For a
circular orbit `r = a`, which collapses to `v = sqrt(GM/r)`.

## 3. Transfer Orbits

A **Hohmann transfer** moves a spacecraft between two circular coplanar orbits using two
burns: one to raise apoapsis to the target radius, a second at apoapsis to circularise. It
is the most propellant-efficient two-burn transfer, but not the fastest.

A **bi-elliptic transfer** uses three burns via a very high intermediate apoapsis. It beats
Hohmann on propellant when the ratio of final to initial radius exceeds roughly 11.94.

## 4. Orbital Elements

Six numbers fix an orbit in space: semi-major axis, eccentricity, inclination, longitude of
the ascending node, argument of periapsis, and true anomaly at epoch.

## 5. Perturbations

Real orbits drift. The dominant term for Earth satellites is `J2`, the oblateness of the
planet, which causes the ascending node to regress and the argument of periapsis to rotate.
Sun-synchronous orbits deliberately exploit `J2` so the orbital plane precesses about one
degree per day, keeping local solar time constant.
