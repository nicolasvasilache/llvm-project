//===- pass-registering-extension-considered-harmful.cpp - Bug repro ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry point for pass-registering-extension-considered-harmful bug repro.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

namespace {

struct DummyPassWithDependentExtension
    : public PassWrapper<DummyPassWithDependentExtension,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyPassWithDependentExtension)

  StringRef getArgument() const final {
    return "dummy-pass-with-dependent-extension";
  }
  StringRef getDescription() const final {
    return "Tests affine data copy utility functions.";
  }
  DummyPassWithDependentExtension() = default;
  DummyPassWithDependentExtension(const DummyPassWithDependentExtension &pass)
      : PassWrapper(pass){};

  void getDependentDialects(DialectRegistry &registry) const override {
    llvm::outs() << "getDependentDialects!\n";
    registry.insert<memref::MemRefDialect>();
    registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *d) {
      llvm::outs() << "add an extension directly to registry\n";
    });
  }
  void runOnOperation() override { llvm::outs() << "Hiya!\n"; }
};

} // namespace

int main(int argc, char **argv) {
  registerAllPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);

  MLIRContext context(registry);
  {
    PassManager pm(&context);
    pm.addPass(std::make_unique<DummyPassWithDependentExtension>());
    (void)pm.run(ModuleOp::create(mlir::UnknownLoc::get(&context)));
  }

  // Simulate pass dynamically created within a pass.
  context.enterMultiThreadedExecution();
  PassManager pm(&context);
  pm.addPass(std::make_unique<DummyPassWithDependentExtension>());
  (void)pm.run(ModuleOp::create(mlir::UnknownLoc::get(&context)));
  context.exitMultiThreadedExecution();

  return 0;
}
