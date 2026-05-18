---
name: New format support
about: Request a reader for a merger-tree format not yet supported.
title: 'Reader: <FORMAT NAME>'
labels: enhancement, reader
assignees: ''
---

## Format

<!-- Name and a one-line description of what produces it
     (which halo finder / tree builder). -->

## Specification

<!-- Link to the format spec, the producing code's documentation,
     or relevant publications. The more detail the better — column
     definitions, units, file layout, byte offsets if binary. -->

## Sample data

<!-- Crucial: is a small sample dataset publicly downloadable, and
     where? Links to ytree's sample collection, the producing
     project's repo, or a personal share all work. Without sample
     data the reader can be drafted but not verified. -->

## Unit conventions

| Quantity | Unit in this format |
|---|---|
| position | <!-- e.g. Mpc/h --> |
| velocity | <!-- e.g. km/s --> |
| mass | <!-- e.g. M_sun/h --> |
| scale factor / redshift | <!-- e.g. listed per snap in a sibling file --> |

## Anything unusual

<!-- Cross-file references? Implicit forest grouping?
     Format-specific defaults that should ship with the reader? -->
