from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn import svm
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from .experiment import Experiment

class QubitClassifier(object):
    """A generic class to classify qubit state.

    Parameters
    ----------
    exp : Experiment
        the Experiment object to do classification with
    qubit : int
        index of the qubit to be classified
    calib_seq : tuple, optional
        the index of the calibration pulses, in the order of
        (|00>, |01>, |10>, |11>), by default None if no
        calibration pulse is included

    Raises
    ------
    TypeError
        if 'exp' is not an Experiment object
    TypeError
        if the experiment is not a single-shot experiment
    ValueError
        the qubit to be classified has out-of-range index
        
    """
    def __init__(self, exp, qubit, calib_seq=None):
        if not isinstance(exp, Experiment):
            raise TypeError('Input is not a Experiment object')

        if exp.scan_setup.loc['Target', 0] != 'Sequencer':
            raise TypeError(
                'Input experiment is not a single-shot experiment')

        if not qubit in range(exp.number_of_readout):
            raise ValueError('Qubit index out of range')

        self._exp = exp
        self._qubit = qubit
        self._calib_seq = calib_seq
        self._clf = self._classify_boundary()

    def _set_assembly(self):
        """Raise error if not implemented in child class."""
        raise NotImplementedError("Not implemented!")

    def _train_set_assembly(self):
        """Raise error if not implemented in child class."""
        raise NotImplementedError("Not implemented!")

    def _seq_concatenate(self):
        """Concatenate 'Sequence' experiments.

        Returns
        -------
        list of list of numpy.ndarray
            concatenated data list
        """
        # find the index of 'Sequence' sweep
        seq_idx = self._exp.scan_setup.isin(['Sequence']).any().to_numpy().nonzero()[0][0]

        # find the index of 'Sequencer' sweep
        # sqcer_idx = self._exp.scan_setup.isin(['Sequencer']).any().to_numpy().nonzero()[0][0]

        # determines number of sequence piece
        self._seq_piece = self._exp.scan_size[seq_idx]

        # initialize concatenated data
        data = [[None for _ in range(2)] for _ in range(len(self._qubit))]

        # concatenate sequence
        for res, quad in product(self._qubit, range(2)):
            data_temp = self._exp.data[res][quad].transpose((2, 0, 1))
            data[res][quad] = data_temp.reshape(-1, self._exp.scan_size[1])

        return data
    
    def _classify_boundary(self, kernel='rbf'):
        """Carry out classification.

        Parameters
        ----------
        kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, optional
            SVM kernel used, by default 'rbf'

        Returns
        -------
        sklearn.estimator
            best estimator determined by sklearn.GridSearchCV
        """
        if self._exp.path.joinpath(f'assets/Projector_Q{self._qubit}.joblib').exists():
            clf = joblib.load(self._exp.path.joinpath(f'assets/Projector_Q{self._qubit}.joblib'))
        else:
            whole_set_scaled, label = self._train_set_assembly()

            # prepare search grid
            C_range = np.logspace(-1.1, 3.1, num=5, base=2)
            gamma_range = np.logspace(-4.1, 1.1, num=6, base=2)
            param_grid = dict(gamma=gamma_range, C=C_range)

            # stratified shuffle for cross-validation
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)

            # do SVM classification 
            grid = GridSearchCV(
                                (svm.SVC(kernel=kernel)),
                                param_grid=param_grid,
                                cv=cv)
            grid.fit(whole_set_scaled, label)

            clf = grid.best_estimator_

            joblib.dump(clf, self._exp.path.joinpath(f'assets/Projector_Q{self._qubit}.joblib'))

        return clf

    def c_matrix(self):
        """Calculate classification confusion matrix.

        Returns
        -------
        numpy.ndarray of shape (n_classes, n_classes)
            Confusion matrix whose i-th row and j-th column entry indicates
            the number of samples with true label being i-th class and
            predicted label being j-th class.
        """
        clf = self._classify_boundary()
        whole_set_scaled, label = self._train_set_assembly()

        predicted_label = clf.predict(whole_set_scaled)

        c_matrix = confusion_matrix(label, predicted_label, normalize='true').T

        return c_matrix

class SingleQubitClassifier(QubitClassifier):

    def __init__(self, exp, qubit, calib_seq=None):
        super().__init__(exp, qubit, calib_seq)

    def _set_assembly(self):
        raw = self._exp.data[self._qubit]
        mean_data = self._exp.average()[self._qubit]
        
        x_center = np.mean(mean_data[0])
        poly_fit = np.polynomial.Polynomial.fit(mean_data[0], mean_data[1], 1)
        y_center = poly_fit(x_center)
        theta = np.pi - np.angle((mean_data[0][0]-x_center)+1j*(poly_fit(mean_data[0][0])-y_center))
        x_rotated = np.cos(theta)*(raw[0]-x_center) - np.sin(theta)*(raw[1]-y_center)
        ground_ratio = np.sum(x_rotated < 0, axis=1)/x_rotated.shape[1]
        idx1 = np.argmax(ground_ratio)
        idx2 = np.argmin(ground_ratio)

        ground_state_hist = np.stack((raw[0][idx1, :], raw[1][idx1, :])).T
        excited_state_hist = np.stack((raw[0][idx2, :], raw[1][idx2, :])).T

        whole_set = np.concatenate((ground_state_hist, excited_state_hist), axis=0)
        set_length = [ground_state_hist.shape[0], excited_state_hist.shape[0]]
        return whole_set, set_length

    def _train_set_assembly(self):
        whole_set, set_length = self._set_assembly()
        whole_set_scaled = minmax_scale(whole_set, feature_range=(-2, 2), axis=0)

        label = np.concatenate((np.full(set_length[0], 0, dtype=int),
                                np.full(set_length[1], 1, dtype=int)))

        return whole_set_scaled, label

    def plot_boundary(self):
        _, set_length = self._set_assembly()
        whole_set_scaled, _ = self._train_set_assembly()

        fidelity = self.fidelity()

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(whole_set_scaled[:set_length[0], 0],
                   whole_set_scaled[:set_length[0], 1],
                   c='r', marker='.', s=20)
        ax.scatter(whole_set_scaled[set_length[0]:, 0],
                   whole_set_scaled[set_length[0]:, 1],
                   c='b', marker='.', s=20)
        ax.scatter(self._clf.support_vectors_[:, 0],
                   self._clf.support_vectors_[:, 1],
                   s=20, linewidth=0.5,
                   facecolors='none', edgecolors='k')

        x_min, x_max = ax.get_xlim()[0], ax.get_xlim()[1]
        y_min, y_max = ax.get_ylim()[0], ax.get_ylim()[1]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                             np.arange(y_min, y_max, 0.005))
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = self._clf.predict(xy).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=[-1, 0, 1, 2], alpha=0.2, colors=['red', 'blue'])
        ax.set_title('Scatter Diagram with the Optimal Readout Boundary\n' +
                     f'Fidelity={fidelity*100:.2f}%', fontsize=16)

        fig.savefig(self._exp.path.joinpath('Single shot measurement and optimal ' +
                                            f'readout boundary on Q{self._qubit} IQ plane.png'))
        plt.show()

    def fidelity(self):
        c_matrix = self.c_matrix()
        infidelity = np.trace(np.fliplr(c_matrix))
        fidelity = 1 - infidelity

        return fidelity

    def _scale_to_train_set(self):
        whole_set, _ = self._set_assembly()
        minimum = np.min(whole_set, axis=0)
        maximum = np.max(whole_set, axis=0)
        scale = [4/(max - min) for (max, min) in zip(maximum, minimum)]
        data_rescaled = []
        for idx, value in enumerate(range(2)):
            res = self._qubit
            quad = value
            data_rescaled.append(self._exp.data[res][quad]*scale[idx] -2 - minimum[idx]*scale[idx])

        return data_rescaled

    def predict(self):
        data_rescaled = self._scale_to_train_set()
        xy = np.vstack([data_rescaled[0].ravel(),
                        data_rescaled[1].ravel()]).T
        prediction = self._clf.predict(xy).reshape(data_rescaled[0].shape)

        return prediction

class TwoQubitClassifier(QubitClassifier):
    # to be implemented
    pass
