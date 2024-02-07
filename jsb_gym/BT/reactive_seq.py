# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)

import py_trees as pt
#import py_trees_ros as ptr

class ReactiveSeq(pt.composites.Composite):
    """
    >>>>>>>>   Copied from the selector code to make into a reactive sequence
    .. graphviz:: dot/selector.dot
    A selector executes each of its child behaviours in turn until one of them
    fails (at which point it itself returns :data:`~py_trees.common.Status.RUNNING` or :data:`~py_trees.common.Status.FAILURE`,
    or it runs out of children at which point it itself returns :data:`~py_trees.common.Status.FAILURE`.
    We usually refer to selecting children as a means of *choosing between priorities*.
    Each child and its subtree represent a decreasingly lower priority path.
    .. note::
       Switching from a low -> high priority branch causes a `stop(INVALID)` signal to be sent to the previously
       executing low priority branch. This signal will percolate down that child's own subtree. Behaviours
       should make sure that they catch this and *destruct* appropriately.
    Make sure you do your appropriate cleanup in the :meth:`terminate()` methods! e.g. cancelling a running goal, or restoring a context.
    .. seealso:: The :ref:`py-trees-demo-selector-program` program demos higher priority switching under a selector.
    Args:
        name (:obj:`str`): the composite behaviour name
        children ([:class:`~py_trees.behaviour.Behaviour`]): list of children to add
    """

    def __init__(self, name="ReactiveSeq", children=None):
        super(ReactiveSeq, self).__init__(name, children)
        self.current_child = None


    def tick(self):
        """
        Run the tick behaviour for this selector. Note that the status
        of the tick is always determined by its children, not
        by the user customised update function.
        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # Required behaviour for *all* behaviours and composites is
        # for tick() to check if it isn't running and initialise
        if self.status != pt.common.Status.RUNNING:
            # selectors dont do anything specific on initialisation
            #   - the current child is managed by the update, never needs to be 'initialised'
            # run subclass (user) handles
            self.initialise()
        # run any work designated by a customised instance of this class
        self.update()
        previous = self.current_child
        for child in self.children:
            for node in child.tick():
                yield node
                if node is child:
                    if node.status == pt.common.Status.RUNNING or node.status == pt.common.Status.FAILURE:
                        self.current_child = child
                        self.status = node.status
                        if previous is None or previous != self.current_child:
                            # we interrupted, invalidate everything at a lower priority
                            passed = False
                            for child in self.children:
                                if passed:
                                    if child.status != pt.common.Status.INVALID:
                                        child.stop(pt.common.Status.INVALID)
                                passed = True if child == self.current_child else passed
                        yield self
                        return
        # all children failed, set failure ourselves and current child to the last bugger who failed us
        # omg i love this comment /Ozer
        self.status = pt.common.Status.FAILURE
        try:
            self.current_child = self.children[-1]
        except IndexError:
            self.current_child = None
        yield self


    def stop(self, new_status=pt.common.Status.INVALID):
        """
        Stopping a selector requires setting the current child to none. Note that it
        is important to implement this here instead of terminate, so users are free
        to subclass this easily with their own terminate and not have to remember
        that they need to call this function manually.
        Args:
            new_status (:class:`~py_trees.common.Status`): the composite is transitioning to this new status
        """
        # retain information about the last running child if the new status is
        # SUCCESS or FAILURE
        if new_status == pt.common.Status.INVALID:
            self.current_child = None
        pt.Composite.stop(self, new_status)


    def __repr__(self):
        """
        Simple string representation of the object.
        Returns:
            :obj:`str`: string representation
        """
        s = "Name       : %s\n" % self.name
        s += "  Status  : %s\n" % self.status
        s += "  Current : %s\n" % (self.current_child.name if self.current_child is not None else "none")
        s += "  Children: %s\n" % [child.name for child in self.children]
        return s
    
    
if __name__ == "__main__":
    RS = ReactiveSeq()