// ivectorbin/ivector-extractor-est.cc

// Copyright 2013  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "util/common-utils.h"
#include "ivector/ivector-extractor.h"
#include "util/kaldi-thread.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;

    const char *usage =
        "Copy binary ivector extractor into txt format\n"
        "Usage: copy-ivectorextractor [options] <model-in> <model-out>\n";

    bool binary = false;
    IvectorExtractorEstimationOptions update_opts;

    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode, default: false");
    po.Register("num-threads", &g_num_threads,
                "Number of threads used in update");

    update_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        model_wxfilename = po.GetArg(2);

    KALDI_LOG << "Reading model";
    IvectorExtractor extractor;
    ReadKaldiObject(model_rxfilename, &extractor);

    WriteKaldiObject(extractor, model_wxfilename, binary);

    KALDI_LOG << "Copied it to "
              << model_wxfilename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


