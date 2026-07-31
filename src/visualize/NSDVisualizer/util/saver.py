def get_cortical_mask(subject, xfmname, type='nearest'):
    """Gets the cortical mask for a particular transform

    Parameters
    ----------
    subject : str
        Subject name
    xfmname : str
        Transform name
    type : str
        Mask type, one of {"cortical", "thin", "thick", "nearest", "line_nearest"}.
          - 'cortical' includes voxels contained within the cortical ribbon, 
          between the freesurfer-estimated white matter and pial surfaces. 
          - 'thin' includes voxels that are < 2mm away from the fiducial surface. 
          - 'thick' includes voxels that are < 8mm away from the fiducial surface.
          - 'nearest' includes only the voxels overlapping the fiducial surface.
          - 'line_nearest' includes all voxels that have any part within the cortical 
            ribbon.

    Returns
    -------
    mask : array
        boolean mask array for cortical voxels in functional space

    Notes
    -----
    "nearest" is a conservative "cortical" mask, while "line_nearest" is a liberal 
    "cortical" mask.
    """
    if type == 'cortical':
        ppts, polys = db.get_surf(subject, "pia", merge=True, nudge=False)
        wpts, polys = db.get_surf(subject, "wm", merge=True, nudge=False)
        thickness = np.sqrt(((ppts - wpts)**2).sum(1))

        dist, idx = get_vox_dist(subject, xfmname)
        cortex = np.zeros(dist.shape, dtype=bool)
        verts = np.unique(idx)
        for i, vert in enumerate(verts):
            mask = idx == vert
            cortex[mask] = dist[mask] <= thickness[vert]
            if i % 100 == 0:
                print("%0.3f%%"%(i/float(len(verts)) * 100))
        return cortex
    elif type in ('thick', 'thin'):
        dist, idx = get_vox_dist(subject, xfmname)
        return dist < dict(thick=8, thin=2)[type]
    else:
        return get_mapper(subject, xfmname, type=type).mask
