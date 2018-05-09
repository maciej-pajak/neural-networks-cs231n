package pl.maciejpajak.network.optimization;

import pl.maciejpajak.network.NetworkConfig;

public enum ParamUpdate {
    SGD {
        @Override
        public Updater getUpdater(UpdaterConfig config) {
            return new Sgd(config);
        }
    },
    MOMENTUM {
        @Override
        public Updater getUpdater(UpdaterConfig config) {
            // TODO
            return null;
        }
    },
    NESEROV_MOMENTUM {
        @Override
        public Updater getUpdater(UpdaterConfig config) {
            // TODO
            return null;
        }
    };

    public abstract Updater getUpdater(UpdaterConfig config);
}
