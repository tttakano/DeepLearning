#!/usr/bin/env python

import argparse

from nltk.translate import bleu_score
import numpy
import progressbar
import six
import sys
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import matplotlib as mpl

#chainer.set_debug(True)

mpl.use('Agg')

UNK = 0
EOS = 1

'''
ex)
xs=[[i, am, taro],[i, was, born, in, japan],[i, love, music],[how,are,you]]
ex=[i, am, taro,i, was, born, in, japan,i, love, music,how,are,you]
exs=[[i, am, taro],[i, was, born, in, japan],[i, love, music],[how,are,you]](embed by neural network)
'''


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


"""            
            hx, cx, _ = self.encoder(None, None, exs)
            _, _, os = self.decoder(hx, cx, eys)
            concat_os = F.concat(os, axis=0)
            concat_ys_out = F.concat(ys_out, axis=0)            
            return F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
"""


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units, batch_size):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            # add
            self.connecter = L.Linear(None, n_units*batch_size)######################################
            # end
            self.W = L.Linear(n_units, n_target_vocab)
            

        self.n_layers = n_layers
        self.n_units = n_units
        # add
        self.prev_hx = chainer.Variable(cuda.cupy.array(numpy.zeros((self.n_layers, 1, self.n_units)), dtype=numpy.float32))
        self.prev_h = chainer.Variable(cuda.cupy.array(numpy.zeros((self.n_layers, 1, self.n_units)), dtype=numpy.float32))
        # end

    def __call__(self, xs, ys):
        xs = [x[::-1] for x in xs]  # reverse input      ["i", "am", "taro"] →["taro", "am", "I"]

        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]  # [eos,y1,y2,...]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]  # [y1,y2,...,eos]

        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)

        hx, cx, _ = self.encoder(None, None, exs)
        #add##################################################################################################################################
        forward_hx = hx[:, :-1]
        sifted_hx = F.concat([self.prev_hx, forward_hx], axis=1)
        in_hx = F.concat((hx, sifted_hx), axis=2)
        out_hx = self.connecter(in_hx)
        out_hx = out_hx.reshape(self.n_layers, -1, self.n_units)
        self.prev_hx = hx[:, -1:]

        '''
        hx[0] = [[0,1,2],[3,4,5],[6,7,8]]                  shape = (layers, batch, units)
        forward_hx[0] = [[0,1,2],[3,4,5]]                  shape = (layers, batch-1, units)
        sifted_hx[0] = [self.prev_hx, [0,1,2],[3,4,5]]     shape = (layers, batch, units)
        in_hx[0] = [hx[0],sifted_hx[0]]                    shape = (layers, batch, units*2)
        out_hx[0] = connecter(in_hx)                       shape = (layers, batch, units)
        '''

        is_start_of_sentence = cuda.cupy.array([1 if word[-1] == 6 else 0 for word in xs]) #6 means word number of * (start of sentece)
        is_start_of_sentence = is_start_of_sentence.reshape(-1, 1)
        
        new_hx = is_start_of_sentence * hx + (1 - is_start_of_sentence) * out_hx

        '''
        When
        hx = [[[1,2,3],[4,5,6],[7,8,9]],[[11,12,13],[14,15,16],[17,18,19]],[21,22,23],[24,25,26],[27,28,29]] (shape = (layer=3, batch=3, unit=3))
        out_hx = [[[10,20,30],[40,50,60],[70,80,90]],[[110,120,130],[140,150,160],[170,180,190]],[210,220,230],[240,250,260],[270,280,290]]
        is_state_of_stence = [[1],[0],[1]] (shape=(batch=3, 1))
        
        Then, 
        new_hx = [[[1,2,3],[4,5,6],[7,8,9]],[[110,120,130],[140,150,160],[170,180,190]],[21,22,23],[24,25,26],[27,28,29]]
        '''

        hx = new_hx
        
        #sys.exit()
        #end############################################################################################################################
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=50):
        batch = len(xs)

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]

            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)

            # add
            forward_h = h[:, :-1]
            sifted_h = F.concat((self.prev_h, forward_h), axis=1)
            in_h = F.concat((h, sifted_h), axis=2)
            out_h = self.connecter(in_h)
            out_h = out_h.reshape(self.n_layers, -1, self.n_units)
            self.prev_h = h[:, -1:]

            is_start_of_sentence = cuda.cupy.array([1 if word[-1] == 6 else 0 for word in xs]) 
            is_start_of_sentence = is_start_of_sentence.reshape(-1, 1)

            new_h = is_start_of_sentence * h + (1 - is_start_of_sentence) * out_h
            h = new_h
            # end
            
            ys = self.xp.full(batch, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype(numpy.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=1, device=-1, max_length=50):  # todo change batch size
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK)
                                 for w in words], numpy.int32)
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--SOURCE', default="./dataset/train.en", help='source sentence list')
    parser.add_argument('--TARGET', default="./dataset/train.jp", help='target sentence list')
    parser.add_argument('--SOURCE_VOCAB', default="./dataset/vocab.en", help='source vocabulary file')
    parser.add_argument('--TARGET_VOCAB', default="./dataset/vocab.jp", help='target vocabulary file')
    parser.add_argument('--validation-source', default="./dataset/test.en",
                        help='source sentence list for validation')
    parser.add_argument('--validation-target', default="./dataset/test.jp",
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--min-source-sentence', type=int, default=0,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=0,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                             'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    args = parser.parse_args()

    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    train_source = load_data(source_ids, args.SOURCE)
    train_target = load_data(target_ids, args.TARGET)
    assert len(train_source) == len(train_target)
    train_data = [(s, t)
                  for s, t in six.moves.zip(train_source, train_target)
                  if args.min_source_sentence <= len(s)
                  <= args.max_source_sentence and
                  args.min_source_sentence <= len(t)
                  <= args.max_source_sentence]
    train_source_unknown = calculate_unknown_ratio(
        [s for s, _ in train_data])
    train_target_unknown = calculate_unknown_ratio(
        [t for _, t in train_data])

    print('Source vocabulary size: %d' % len(source_ids))
    print('Target vocabulary size: %d' % len(target_ids))
    print('Train data size: %d' % len(train_data))
    print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit, args.batchsize)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize, True, False)  # shuffle=false
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'validation/main/bleu',
         'elapsed_time']),
        trigger=(args.log_interval, 'iteration'))
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validationl/main/loss'], x_key='epoch', file_name='loss.png'))  # to plot
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))

    if args.validation_source and args.validation_target:
        test_source = load_data(source_ids, args.validation_source)
        test_target = load_data(target_ids, args.validation_target)
        assert len(test_source) == len(test_target)
        test_data = list(six.moves.zip(test_source, test_target))
        test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
        test_source_unknown = calculate_unknown_ratio(
            [s for s, _ in test_data])
        test_target_unknown = calculate_unknown_ratio(
            [t for _, t in test_data])

        print('Validation data: %d' % len(test_data))
        print('Validation source unknown ratio: %.2f%%' %
              (test_source_unknown * 100))
        print('Validation target unknown ratio: %.2f%%' %
              (test_target_unknown * 100))

        @chainer.training.make_extension()  # per 1 epoch
        def translate(trainer):
            source, target = test_data[numpy.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            source_sentence = ' '.join([source_words[x] for x in source])
            target_sentence = ' '.join([target_words[y] for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            print('# source : ' + source_sentence)
            print('#  result : ' + result_sentence)
            print('#  expect : ' + target_sentence)

        trainer.extend(
            translate, trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateBleu(
                model, test_data, 'validation/main/bleu', device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))

    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()


