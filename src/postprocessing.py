import matplotlib.pyplot as plt

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


class Postprocessing:
    """
    This class is used to postprocess the results of the scenarios. This should work
    for an arbitrary number of scenarios. It should be possible to compare the scenarios
    and generate a report of the results.

    Arguments:
        scenarios (list): List of scenarios to compare. Each scenario should be an instance
        of PyTradeShifts.

    Returns:
        None
    """

    def __init__(self, scenarios):
        self.scenarios = scenarios
        self._calculate_stuff()

    def _calculate_stuff(self):
        """
        Hidden method to calculate stuff, like comparing scenarios

        """
        pass

    def plot(self):
        """
        Plots the results of the scenarios. This could be something like comparing on the
        world map where the scenarios differ or a visual comparison of the stability of
        the graphs of the scenarions.

        Not sure if this needs to be a method or could also just be in report.
        """
        pass

    def report(self):
        """
        This method generates a report of the results of the scenarios. This could be a
        pdf or a markdown file. This should contain the results of the scenarios and
        the comparison of the scenarios, like with plots or tables.
        """

        pass
