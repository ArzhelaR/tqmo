"""
Mesh Convolution Module
========================

This module provides functions to extract topological and geometrical features from quad meshes
and convert them into feature matrices (templates) for the observation.

Functions:
    get_x: Extract and select the most important dart features for the observation.
    get_template: Extract topological features based on neighborhood depth (nodes irregularities). Final matrix size is (n_darts_selected, deep).
    get_template_new: Extract topological features with additional geometrical information (nodes irregularities, geometric quality and is_starred). Final matrix size is (n_darts_selected, deep+2).
    get_template_boundary: Extract nodes irregularities with nodes boundary flags. Final matrix size is (n_darts_selected, deep*2).
    get_template_deg: Extract nodes irregularities with nodes degree information. Final matrix size is (n_darts_selected, deep*2).
"""

import numpy as np
from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from mesh_model.mesh_struct.mesh_elements import Dart
from view.mesh_plotter.mesh_plots import plot_mesh


def get_x(m_analysis, n_darts_selected: int, deep :int, analysis_type, restricted:bool):
    """
    Get the feature matrix for the observation.

    This function extracts topological and geometric features from the mesh based on the specified analysis type,
    then selects the n_darts_selected darts with the highest irregularities.

    Args:
        m_analysis (QuadMeshTopoAnalysis): Mesh analysis object containing the mesh 2-cmap structure.
        n_darts_selected (int): Number of darts to select from the template.
        deep (int): Depth of neighborhood.
        analysis_type (str): Type of analysis to use: "boundary", "old", "new", or "topo".
                            - "topo": Default topological analysis
                            - "boundary": Include boundary node flags
                            - "new": Topological features with geometric information (quality, starred)
                            - "old": Legacy topological features only
        restricted (bool): Whether to restrict darts to valid action positions (deprecated).

    Returns:
        tuple: (X, valid_dart_ids)
            - X (np.ndarray): The observation matrix
            - valid_dart_ids (np.ndarray): IDs of selected darts, padded with -1 if fewer than n_darts_selected
    """
    mesh = m_analysis.mesh
    if analysis_type == "boundary":
        template, darts_id = get_template_boundary(m_analysis, deep)
    elif analysis_type == "old":
        template, darts_id = get_template(m_analysis, deep)
    elif analysis_type == "new":
        template, darts_id = get_template_new(m_analysis, deep)
    elif analysis_type == "topo": # the last method implemented and the one mainly used
        template, darts_id = get_template(m_analysis, deep)
    else :
        raise ValueError("Unknown analysis type")

    # if degree:
    #     deep = int(deep / 2)
    #     template, darts_id = get_template_boundary(m_analysis, deep)
    # else:
    #     template, darts_id = get_template(m_analysis, deep)

    if restricted: # we don't use it anymore
        darts_to_delete = []
        darts_id = []
        for i, d_info in enumerate(mesh.active_darts()):
            d_id = d_info[0]
            if d_info[2] == -1 or not m_analysis.isValidAction(d_info[0], 4)[0]:  # test the validity of all action type
                darts_to_delete.append(i)
            else:
                darts_id.append(d_id)
        valid_template = np.delete(template, darts_to_delete, axis=0)
    else:
        valid_template = template
    score_sum = np.sum(np.abs(valid_template[:,:deep]), axis=1)
    indices_selected_darts = np.argsort(score_sum)[-n_darts_selected:][::-1]

    valid_dart_ids = [darts_id[i] for i in indices_selected_darts]
    X = valid_template[indices_selected_darts, :]
    while len(valid_dart_ids) != n_darts_selected:
        valid_dart_ids.append(-1)
        X = np.vstack((X, np.zeros((1, X.shape[1]))))
    return X, np.array(valid_dart_ids)


def get_template(m_analysis, deep: int):
    """
    Extract topological features from the mesh based on neighborhood depth.

    For each active dart, this function extracts the topological scores of nodes in its neighborhood
    up to a specified depth. The neighborhood is defined using the combinatorial map's beta operations.

    Algorithm:
        1. For each dart d, extract the 4 corner nodes (A, B, C, D) using beta-1 operations
        2. For deep > 4, expand the neighborhood by traversing adjacent faces (beta-2 operations)
        3. Continue until 'deep' number of nodes are collected or the mesh boundary is reached
        4. Store the score of each node in the template matrix

    Args:
        m_analysis (QuadMeshTopoAnalysis): Mesh analysis object.
        deep (int): Depth of neighborhood expansion (ie. number of nodes retrieved). Ex: 4,12,36, etc.

    Returns:
        tuple: (template, dart_ids)
            - template (np.ndarray): Feature matrix of shape (n_active_darts, deep) containing node scores.
            - dart_ids (list[int]): List of active dart IDs corresponding to template rows.
    """
    size = len(m_analysis.mesh.dart_info)
    template = np.zeros((size, deep), dtype=np.int64)
    dart_ids = []
    n_darts = 0

    for d_info in m_analysis.mesh.active_darts():
        n_darts += 1
        d_id = d_info[0]
        dart_ids.append(d_id)
        d = Dart(m_analysis.mesh, d_id)
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()
        d111 = d11.get_beta(1)
        D = d111.get_node()

        # Template niveau 1
        template[n_darts - 1, 0] = A.get_score()
        template[n_darts - 1, 1] = B.get_score()
        template[n_darts - 1, 2] = C.get_score()
        template[n_darts - 1, 3] = D.get_score()

        E = [A,B,C,D]
        deep_captured = len(E)
        d2 = d.get_beta(2)
        d12 = d1.get_beta(2)
        d112 = d11.get_beta(2)
        d1112 = d111.get_beta(2)
        F = [d2, d12, d112, d1112]
        if deep>4:
            while len(E)<deep:
                df = F.pop(0)
                if df is not None:
                    df1 = df.get_beta(1)
                    df11 = df1.get_beta(1)
                    df111 = df11.get_beta(1)
                    F.append(df1)
                    F.append(df11)
                    F.append(df111)
                    N1, N2 = df11.get_node(), df111.get_node()
                    E.append(N1)
                    template[n_darts-1, len(E)-1] = N1.get_score()
                    E.append(N2)
                    template[n_darts - 1, len(E)-1] = N2.get_score()
                else:
                    E.extend([None,None])
                    F.append(None)
                    F.append(None)
                    F.append(None)
                    #template[n_darts - 1, len(E) - 1] = -500 # dummy vertices are assigned to -500
                    #template[n_darts - 1, len(E) - 2] = -500 # dummy vertices are assigned to -500

    template = template[:n_darts, :]

    return template, dart_ids

def get_template_new(m_analysis, deep: int):
    """
    Extract topological features with additional geometrical information.

    This is an enhanced version of get_template that includes geometric observations for each dart.

    Algorithm:
        1. Same as get_template for extracting topological node scores
        2. Additionally extract geometric features: dart quality and starred flag
        3. Store geometric features in the last 2 columns of the template matrix

    Args:
        m_analysis (QuadMeshTopoAnalysis): Mesh analysis object.
        deep (int): Depth of neighborhood expansion.

    Returns:
        tuple: (template, dart_ids)
            - template (np.ndarray): Feature matrix of shape (n_active_darts, deep+2) containing:
                * Columns 0 to deep-1: Node topological scores
                * Column deep: Dart quality (geometric feature)
                * Column deep+1: Dart starred flag (binary indicator)
            - dart_ids (list[int]): List of active dart IDs corresponding to template rows.

    Differences from get_template:
        - Template size is (deep+2) instead of (deep)
        - Includes dart quality and starred status for better feature representation
        - Better for models trained on enriched geometric information
    """
    size = len(m_analysis.mesh.dart_info)
    template = np.zeros((size, deep+2), dtype=np.int64)
    dart_ids = []
    n_darts = 0

    for d_info in m_analysis.mesh.active_darts():
        n_darts += 1
        d_id = d_info[0]
        dart_ids.append(d_id)
        d = Dart(m_analysis.mesh, d_id)
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()
        d111 = d11.get_beta(1)
        D = d111.get_node()

        # Geometrical observation at the end of the matrix
        template[n_darts - 1, deep] = d.get_quality()
        template[n_darts - 1, deep + 1] = d.is_starred()


        # Template niveau 1
        template[n_darts - 1, 0] = A.get_score()
        template[n_darts - 1, 1] = B.get_score()
        template[n_darts - 1, 2] = C.get_score()
        template[n_darts - 1, 3] = D.get_score()


        E = [A,B,C,D]
        deep_captured = len(E)
        d2 = d.get_beta(2)
        d12 = d1.get_beta(2)
        d112 = d11.get_beta(2)
        d1112 = d111.get_beta(2)
        F = [d2, d12, d112, d1112]
        if deep>4:
            while len(E)<deep:
                df = F.pop(0)
                if df is not None:
                    df1 = df.get_beta(1)
                    df11 = df1.get_beta(1)
                    df111 = df11.get_beta(1)
                    F.append(df1)
                    F.append(df11)
                    F.append(df111)
                    N1, N2 = df11.get_node(), df111.get_node()
                    E.append(N1)
                    template[n_darts-1, len(E)-1] = N1.get_score()
                    E.append(N2)
                    template[n_darts - 1, len(E)-1] = N2.get_score()
                else:
                    E.extend([None,None])
                    F.extend([None,None,None])
                    #template[n_darts - 1, len(E) - 1] = -500 # dummy vertices are assigned to -500
                    #template[n_darts - 1, len(E) - 2] = -500 # dummy vertices are assigned to -500

    template = template[:n_darts, :]

    return template, dart_ids

def get_template_boundary(m_analysis, deep: int):
    """
    Extract topological features with boundary flags for each node.
    Boundary information is crucial for mesh operations that may be restricted on edges.

    Algorithm:
        1. Same topological neighborhood extraction as get_template
        2. Additionally extract boundary status (get_bdy_flag()) for each node

    Args:
        m_analysis (QuadMeshTopoAnalysis): Mesh analysis object.
        deep (int): Depth of neighborhood expansion.

    Returns:
        tuple: (template, dart_ids)
            - template (np.ndarray): Feature matrix of shape (n_active_darts, deep*2) containing:
                * Columns 0 to deep-1: Node topological scores
                * Columns deep to 2*deep-1: Node boundary flags (0 if on corner, 1 if on edge, 2 if internal, -1 if ghost)
            - dart_ids (list[int]): List of active dart IDs corresponding to template rows.
    """
    size = len(m_analysis.mesh.dart_info)
    template = np.zeros((size, deep*2), dtype=np.int64)
    dart_ids = []
    n_darts = 0

    for d_info in m_analysis.mesh.active_darts():
        n_darts += 1
        d_id = d_info[0]
        dart_ids.append(d_id)
        d = Dart(m_analysis.mesh, d_id)
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()
        d111 = d11.get_beta(1)
        D = d111.get_node()

        # Template niveau 1
        template[n_darts - 1, 0] = A.get_score()
        template[n_darts - 1, deep] = A.get_bdy_flag()
        template[n_darts - 1, 1] = B.get_score()
        template[n_darts - 1, deep+1] = B.get_bdy_flag()
        template[n_darts - 1, 2] = C.get_score()
        template[n_darts - 1, deep+2] = C.get_bdy_flag()
        template[n_darts - 1, 3] = D.get_score()
        template[n_darts - 1, deep + 3] = D.get_bdy_flag()

        E = [A, B, C, D]
        deep_captured = len(E)
        d2 = d.get_beta(2)
        d12 = d1.get_beta(2)
        d112 = d11.get_beta(2)
        d1112 = d111.get_beta(2)
        F = [d2, d12, d112, d1112]
        if deep > 4:
            while len(E) < deep:
                if len(F)<= 1:
                    plot_mesh(m_analysis.mesh, debug=True)
                    raise ValueError("empty list")
                df = F.pop(0)
                if df is not None:
                    df1 = df.get_beta(1)
                    df11 = df1.get_beta(1)
                    df111 = df11.get_beta(1)
                    F.append(df1)
                    F.append(df11)
                    F.append(df111)
                    N1, N2 = df11.get_node(), df111.get_node()
                    E.append(N1)
                    template[n_darts-1, len(E)-1] = N1.get_score()
                    template[n_darts-1, deep + len(E)-1] = N1.get_bdy_flag()
                    E.append(N2)
                    template[n_darts - 1, len(E)-1] = N2.get_score()
                    template[n_darts - 1, deep + len(E)-1] = N2.get_bdy_flag()
                else:
                    E.extend([None,None])
                    F.extend([None,None,None])
                    # ghost nodes are flagged to -1
                    template[n_darts - 1, deep + len(E)-1] = -1
                    template[n_darts - 1, deep + len(E)-2] = -1
                    #template[n_darts - 1, len(E) - 1] = -500 # dummy vertices are assigned to -500
                    #template[n_darts - 1, len(E) - 2] = -500 # dummy vertices are assigned to -500

    template = template[:n_darts, :]
    return template, dart_ids


def get_template_deg(m_analysis, deep: int, nodes_scores, nodes_adjacency):
    """
    Extract topological features with node degree (adjacency) information.

    Algorithm:
        1. Same topological neighborhood extraction as get_template
        2. Additionally extract node adjacency count from nodes_adjacency mapping

    Args:
        m_analysis (QuadMeshTopoAnalysis): Mesh analysis object.
        deep (int): Depth of neighborhood expansion.
        nodes_scores (list[int]): List of node scores (preparation for future use).
        nodes_adjacency (dict or array): Mapping of node ID to adjacency count (node degree).

    Returns:
        tuple: (template, dart_ids)
            - template (np.ndarray): Feature matrix of shape (n_active_darts, deep*2) containing:
                * Columns 0 to deep-1: Node topological scores
                * Columns deep to 2*deep-1: Node adjacency counts (degree)
            - dart_ids (list[int]): List of active dart IDs corresponding to template rows.
    """
    size = len(m_analysis.mesh.dart_info)
    template = np.zeros((size, deep*2), dtype=np.int64)
    dart_ids = []
    n_darts = 0

    for d_info in m_analysis.mesh.active_darts():
        n_darts += 1
        d_id = d_info[0]
        dart_ids.append(d_id)
        d = Dart(m_analysis.mesh, d_id)
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()
        d111 = d11.get_beta(1)
        D = d111.get_node()

        # Template niveau 1
        template[n_darts - 1, 0] = A.get_score()
        template[n_darts - 1, deep] = nodes_adjacency[A.id]
        template[n_darts - 1, 1] = B.get_score()
        template[n_darts - 1, deep+1] = nodes_adjacency[B.id]
        template[n_darts - 1, 2] = C.get_score()
        template[n_darts - 1, deep+2] = nodes_adjacency[C.id]
        template[n_darts - 1, 3] = D.get_score()
        template[n_darts - 1, deep + 3] = nodes_adjacency[D.id]

        E = [A, B, C, D]
        deep_captured = len(E)
        d2 = d.get_beta(2)
        d12 = d1.get_beta(2)
        d112 = d11.get_beta(2)
        d1112 = d111.get_beta(2)
        F = [d2, d12, d112, d1112]
        if deep > 4:
            while len(E) < deep:
                df = F.pop(0)
                if df is not None:
                    df1 = df.get_beta(1)
                    df11 = df1.get_beta(1)
                    df111 = df11.get_beta(1)
                    F.append(df1)
                    F.append(df11)
                    F.append(df111)
                    N1, N2 = df11.get_node(), df111.get_node()
                    E.append(N1)
                    template[n_darts-1, len(E)-1] = N1.get_score()
                    template[n_darts-1, deep + len(E)-1] = nodes_adjacency[N1.id]
                    E.append(N2)
                    template[n_darts - 1, len(E)-1] = N2.get_score()
                    template[n_darts - 1, deep + len(E)-1] = nodes_adjacency[N2.id]
                else:
                    E.extend([None,None])
                    F.extend([None,None,None])
                    #template[n_darts - 1, len(E) - 1] = -500 # dummy vertices are assigned to -500
                    #template[n_darts - 1, len(E) - 2] = -500 # dummy vertices are assigned to -500

    template = template[:n_darts, :]
    return template, dart_ids