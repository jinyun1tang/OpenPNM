# -*- coding: utf-8 -*-
"""
===============================================================================
graph_theory_algorithms: Two classes for graph theory union/find operations
===============================================================================

"""
import scipy as sp


class quick_union_algs:
    r"""
    The graph theory quick union (dynamic) algorithms: Find root, quick union,
    weighted quick union, and weighted quick-union with path compression.
    In OpenPNM, graph vertices correspond to the pores and the graph to the
    pore network. In graph theory, directed weighted graphs are sometimes
    called networks, not to be confused with the OpenPNM pore networks where
    the edges (or throats) are not directed.
    """

    def __init__(self, parents_array):
        r"""
        Initialize the (self) instance attributes.
        
        Parameters
        ----------
        parents_array : array_like
            An array containing the parent's id value (integer) of every
            vertex in the graph. The array indices correspond to the vertices.

        Examples
        --------
        The following graph, made of 11 vertices and 3 connected
        components, is considered:
            (0)       (3)      (7)
             /\        /\
            /  \      /  \
          (1)  (2)  (4)  (5)
                     |
                    (9)
                     /\
                    /  \
                  (6)  (8)
                        |
                       (10)
        >>> from graph_theory_algorithms import quick_union_algs
        >>> import scipy as sp
        >>> graph=sp.array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind=quick_union_algs(graph)
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
      
        Notes
        -----
        self.id_updated array is updated (if applicable) each time one of the
        methods self._root_path_compression, self.build_quick_union, and/or
        self.build_weighted_quick_union is ran.
        """
        # User defined parents ids values
        #self.id_original = sp.array((parents_array),dtype=int)

        # Updated ids
        self.id_updated = sp.array((parents_array),dtype=int)

    def _root(self, v_i):
        r"""
        Find the roots of an array of vertices
        (pores) without path compression.
        
        Parameters
        ----------
        v_i : array_like
            An array of integers corresponding to the vertices (pores)
            whose root is sought.

        Returns
        -------
        Returns an array of integers corresponding to the roots of
        of vertices v_i.

        Examples
        --------
        The following graph, made of 11 vertices and 3 connected
        components, is considered:
            (0)       (3)      (7)
             /\        /\
            /  \      /  \
          (1)  (2)  (4)  (5)
                     |
                    (9)
                     /\
                    /  \
                  (6)  (8)
                        |
                       (10)        
        >>> from graph_theory_algorithms import quick_union_algs
        >>> import scipy as sp
        >>> graph=sp.array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind=quick_union_algs(graph)
        >>> graphUnionFind._root([2,10,7])
        array([0, 3, 7])
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        """
        # chase the parent of the parent until reaching the root
        while ( sp.any(v_i != self.id_updated[v_i]) ):
            v_i = self.id_updated[v_i]
        return v_i
    
    def _root_path_compression(self, v_i, pc_type='path_halving'):
        r"""
        Find the roots of an array of vertices (pores) with path compression.
        Path compression consists in updating the self.id_updated array
        elements within the covered path from element to test (i) to
        its root. The update consists in making vertices pointing to their
        respective grandparents or roots (depending on the path compression
        type) while they were, initially, pointing to their parents.

        Parameters
        ----------
        v_i : array_like
            An array of integers corresponding to the vertices (pores)
            whose root is sought.
        pc_type : str, optional
            Use path_halving path compression type when searching for the root
            to set the self.id_updated of each examined vertex (pore) to its
            grandparent. With pc_type='full_pc', the self.id_updated of each
            examined vertex is set to its root.

        Returns
        -------
        Returns an array of integers corresponding to the roots of
        of vertices v_i.

        Examples
        --------
        The following graph, made of 11 vertices and 3 connected
        components, is considered:
            (0)       (3)      (7)
             /\        /\
            /  \      /  \
          (1)  (2)  (4)  (5)
                     |
                    (9)
                     /\
                    /  \
                  (6)  (8)
                        |
                       (10)
        >>> from graph_theory_algorithms import quick_union_algs
        >>> import scipy as sp
        >>> graph=sp.array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind=quick_union_algs(graph)
        >>> graphUnionFind._root([2,10,7])
        array([0, 3, 7])
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 9, 3, 9])
        >>> graphUnionFind=quick_union_algs(graph)
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind._root_path_compression([2,10,7],pc_type='full_pc')
        array([0, 3, 7])
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 3, 3, 3])

        Notes
        -----
        The path_halving path compression type is a simpler one-pass variant
        of the full path compression. The latter requires a second loop in
        the self._root_path_compression method to set the self.id_updated of
        each examined vertex (pore) to its root.
        """
        v_j = v_i

        if (pc_type == 'path_halving'):
            # chase the parent of the parent until reaching the root
            while ( sp.any(v_j != self.id_updated[v_j]) ):
                self.id_updated[v_j] = self.id_updated[self.id_updated[v_j]]
                v_j = self.id_updated[v_j]

        elif (pc_type == 'full_pc'):
            # chase the parent of the parent until reaching the root
            while ( sp.any(v_j != self.id_updated[v_j]) ):
                v_j = self.id_updated[v_j]
    
            while ( sp.any(v_i != v_j) ):
                v_j_new = self.id_updated[v_i]
                self.id_updated[v_i] = v_j
                v_i = v_j_new
        return v_j

    def find_root(self, elements_to_test=[], path_compression=True, \
                  path_compression_type='path_halving'):
        r"""
        Find the roots of an array of vertices (pores) and update their
        corresponding ids in the self.id_updated array with the roots values.
        This method can be used to check the connectivity between different
        pores and whether or not they belong to the same connected component
        of the graph. Vertices (pores) with the same root are within the same
        graph connected component. They are, thus, connected.

        Parameters
        ----------
        elements_to_test : array_like
            An array containing the id values (integers) of
            the graph vertices whose root is sought.
        path_compression : boolean, optional
            Enable path compression when searching for the root of graph
            vertices, the default is True.
        path_compression_type : str, optional
            If path_compression is True, use path_halving path compression
            type when searching for the root to set the self.id_updated of
            each examined vertex (pore) to its grandparent. With
            pc_type='full_pc', the self.id_updated of each examined vertex
            is set to its root.

        Examples
        --------
        The following graph, made of 11 vertices and 3 connected
        components, is considered:
            (0)       (3)      (7)
             /\        /\
            /  \      /  \
          (1)  (2)  (4)  (5)
                     |
                    (9)
                     /\
                    /  \
                  (6)  (8)
                        |
                       (10)
        >>> from graph_theory_algorithms import quick_union_algs
        >>> import scipy as sp
        >>> graph=sp.array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind=quick_union_algs(graph)
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind.find_root([2,10,7])
        >>> graphUnionFind.id_updated[sp.array([2,10,7])]
        array([0, 3, 7])
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 9, 3, 3])

        Notes
        -----
        In addition to finding the roots, this method also updates the
        self.id_updated array elements within the covered path from
        elements_to_test array elements to their respective roots if
        path_compression is True.
        """
        if (path_compression):
            self.id_updated[elements_to_test]= \
            self._root_path_compression(elements_to_test, \
                                        pc_type=path_compression_type)
        else:
            self.id_updated[elements_to_test]= \
            self._root(elements_to_test)

    def build_quick_union(self, minor, main, path_compression=True, \
                          path_compression_type='path_halving'):
        r"""
        Build unions between connected components whose minor and main
        arrays elements are, respectively, one of their vertices (pores).
        In this method, the union between minor and main vertices is
        performed by creating edges (throats) between their roots to
        keep the graph flat. The parents array (self.id_updated) is then
        updated by making the roots of the minor vertices (pores) pointing
        to the roots of the main vertices.

        Parameters
        ----------
        minor : array_like
            An array containing the indices (integers) of first vertices
            through which the union operations are performed.
        main : array_like
            An array corresponding to the second vertices (pores).
        path_compression : boolean, optional
            Enable path compression when searching the root of the graph
            vertices, the default is True.
        path_compression_type : str, optional
            If path_compression is True, use path_halving path compression
            type when searching for the root to set the self.id_updated of
            each examined vertex (pore) to its grandparent. With
            pc_type='full_pc', the self.id_updated of each examined vertex
            is set to its root.

        Examples
        --------
        The following graph, made of 11 vertices and 3 connected
        components, is considered:
            (0)       (3)      (7)
             /\        /\
            /  \      /  \
          (1)  (2)  (4)  (5)
                     |
                    (9)
                     /\
                    /  \
                  (6)  (8)
                        |
                       (10)
        >>> from graph_theory_algorithms import quick_union_algs
        >>> import scipy as sp
        >>> graph=sp.array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind=quick_union_algs(graph)
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind.build_quick_union(sp.array([2,0]),sp.array([7,6]))
        >>> graphUnionFind.id_updated
        array([3, 0, 0, 3, 3, 3, 4, 7, 9, 4, 8])

        Notes
        -----
        If a union is performed using the build_quick_union method,
        the self.size array (if already defined) is not updated.
        """
        if (path_compression):
            i = self._root_path_compression(minor, \
                                            pc_type=path_compression_type)
            j = self._root_path_compression(main, \
                                            pc_type=path_compression_type)
        else:
            i = self._root(minor)
            j = self._root(main)

        if (sp.all(i == j)):
            return

        else:
            # i is no longer a root. It's parent (also root) is j
            self.id_updated[i] = j

    def build_weighted_quick_union(self, minor, main, path_compression=True, \
                                   path_compression_type='path_halving'):
        r"""
        Build a union between two connected components whose minor and
        main are, respectively, one of their vertices (pores).
        In this method, the union between minor and main vertices is
        performed by creating an edge (throat) between their roots to
        keep the graph flat. A weighting based on the two connected
        components sizes (number of vertices or pores) is used to decide
        which root will be kept as root and which one will be
        a descendant.
        
        Parameters
        ----------
        minor : integer
            An integer corresponding to the first vertex through which
            the union operation is performed.
        main : integer
            An integer corresponding to the second vertex (pore).
        path_compression : boolean, optional
            Enable path compression when searching the root of graph
            vertices, the default is True.
        path_compression_type : str, optional
            If path_compression is True, use path_halving path compression
            type when searching for the root to set the self.id_updated of
            each examined vertex (pore) to its grandparent. With
            pc_type='full_pc', the self.id_updated of each examined vertex
            is set to its root.
        
        Returns
        -------
        Updates the parents array (self.id_updated).
        Updates or creates (if not yet existing) the roots and the
        connected components size arrays.

        Examples
        --------
        The following graph, made of 11 vertices and 3 connected
        components, is considered:
            (0)       (3)      (7)
             /\        /\
            /  \      /  \
          (1)  (2)  (4)  (5)
                     |
                    (9)
                     /\
                    /  \
                  (6)  (8)
                        |
                       (10)
        >>> from graph_theory_algorithms import quick_union_algs
        >>> import scipy as sp
        >>> graph=sp.array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind=quick_union_algs(graph)
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 9, 7, 9, 4, 8])
        >>> graphUnionFind.build_weighted_quick_union(2,7)
        >>> graphUnionFind.id_updated
        array([0, 0, 0, 3, 3, 3, 3, 0, 3, 3, 3])
        >>> graphUnionFind.roots
        array([0, 0, 0, 3, 3, 3, 3, 0, 3, 3, 3])
        >>> graphUnionFind.size
        array([[0, 3],
               [4, 7]])
        """
        # Check the type of the given inputs
        if ((type(minor)!=int) or (type(main)!=int)):
            raise Exception('Received ids must be of type int')

        # Find the roots of the input vertices (pores)
        if (path_compression):
            i = self._root_path_compression(minor, \
                                            pc_type=path_compression_type)
            j = self._root_path_compression(main, \
                                            pc_type=path_compression_type)
        else:
            i = self._root(minor)
            j = self._root(main)

        if i == j:
            return

        try:
            # Check if the self.size array exists
            self.size
        except:
            # If the slef.size array does not exist, build it
            # The self.roots array is needed to build self.size
            # Thus, check if self.roots exists, otherwise, build it too
            try:
                self.roots
                if ( len(self.roots) != len(self.id_updated) ):
                    raise Exception
            except:
                self.find_root( elements_to_test = \
                               sp.arange(len(self.id_updated), dtype=int), \
                               path_compression=path_compression, \
                               pc_type=path_compression_type)
                self.roots=self.id_updated
            self.size=sp.array(sp.unique(self.roots,return_counts=True))

        # Index of the root i in the self.size array
        i_index=int(sp.where((self.size[0]==i))[0])
        # Size of the connected component rooted at i
        i_size = self.size[1][i_index]
        
        # Index of the root j in the self.size array
        j_index=int(sp.where((self.size[0]==j))[0])
        # Size of the connected component rooted at j
        j_size=(self.size)[1][j_index]

        if (i_size <= j_size):
            # Keep root j

            # i is no longer a root. Its parent (also root) is now j
            self.id_updated[i] = j
            # Update self.roots
            self.roots[ self.roots == i ] = j

            # Update self.size
            self.size[1,j_index] += self.size[1,i_index]
            self.size = sp.delete(self.size,i_index,axis=1)
        else:
            # Keep root i

            # j is no longer a root. Its parent (also root) is now i
            self.id_updated[j] = i
            # Update self.roots
            self.roots[ self.roots == j ] = i

            # Update self.size
            self.size[1,i_index] += self.size[1,j_index]
            self.size = sp.delete(self.size,j_index,axis=1)


class depth_first_search_alg:
    r"""Find roots using the depth first-search method in order to check the
    connectivity between the different graph components. This implementation
    uses as an argument the adjacency list that is recommended for sparse
    networks. The use of the adjacency matrix (not available on this class)
    is more suitable for dense graphs.
    """

    def __init__(self, adjacency_list):
        self.adj_list= sp.array( \
        [sp.array(adjacency_list[i],dtype=int) \
         for i in range(len(adjacency_list))])

    def _visit_for_depth_first_search(self,val,now,k):
        self.roots[k]=now
        for x in (self.adj_list[k]):
            if (val[x]==-1):
                self._visit_for_depth_first_search(self.roots,now,k=x)

    def depth_first_search(self):
        r"""
        Check the connectivity between the different graph components.
        
        Returns
        -------
        self.roots : array_like
            An array containing labels for each of the connected components.
        
        Examples
        --------
        The following graph, made of 11 vertices and 3 connected
        components, is considered:
            (0)       (3)      (7)
             /\        /\
            /  \      /  \
          (1)  (2)  (4)  (5)
                     |
                    (9)
                     /\
                    /  \
                  (6)  (8)
                        |
                       (10)
        >>> from graph_theory_algorithms import depth_first_search_alg
        >>> import scipy as sp
        >>> graph=sp.array([ sp.array([1,2]),
                         sp.array([0]),
                         sp.array([0]),
                         sp.array([4,5]),
                         sp.array([3,9]),
                         sp.array([3]),
                         sp.array([9]),
                         sp.array([]),
                         sp.array([9,10]),
                         sp.array([4,6,8]),
                         sp.array([8])])
        >>> dfs=depth_first_search_alg(graph)
        >>> dfs.depth_first_search()
        array([0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1])
        
        Notes
        -----
        An improved version of the depth first search algorithm (Pearce, 2005)
        is available on scipy; scipy.sparse.csgraph.connected_components.
        """
        now=-1
        self.roots=-sp.ones((len(self.adj_list)),dtype=int)
        for i in range(len(self.adj_list)):
            if (self.roots[i]==-1):
                now+=1
                self._visit_for_depth_first_search(self.roots,now,k=i)      
        return self.roots
