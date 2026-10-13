# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Context scoping tests for Portkey AI streaming instrumentation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace

import pytest

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.portkey.wrappers import (
    AsyncPortkeyStreamWrapper,
    PortkeyStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler


def _chunk(content: str, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="stream-chunk-1",
        model="gpt-4o",
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ending", ["drain", "close", "close_early", "error"])
async def test_stream_activates_only_during_reads(
    tracer_provider,
    span_exporter,
    caplog,
    asynchronous: bool,
    ending: str,
) -> None:
    before = context.get_current()
    invocation = TelemetryHandler(
        tracer_provider=tracer_provider,
    ).inference("portkey", request_model="test-model")
    error = ConnectionError("stream failed")
    chunk = _chunk("Hello ", "stop")

    def produce() -> Iterator[SimpleNamespace]:
        assert (
            trace.get_current_span().get_span_context()
            == trace.get_current_span(invocation.context).get_span_context()
        )
        yield chunk
        assert (
            trace.get_current_span().get_span_context()
            == trace.get_current_span(invocation.context).get_span_context()
        )
        if ending == "error":
            raise error

    async def aproduce() -> AsyncIterator[SimpleNamespace]:
        for item in produce():
            await asyncio.sleep(0)
            yield item

    if asynchronous:
        stream = AsyncPortkeyStreamWrapper(aproduce(), invocation, False)
    else:
        stream = PortkeyStreamWrapper(produce(), invocation, False)
    # The stream sits in the caller's context unconsumed: the invocation must
    # not be current here.
    assert context.get_current() is before

    if ending != "close_early":
        if asynchronous:
            assert await asyncio.create_task(anext(stream)) is chunk
        else:
            assert next(stream) is chunk
        assert context.get_current() is before
        assert span_exporter.get_finished_spans() == ()

    if ending in ("close", "close_early"):
        if asynchronous:
            await stream.aclose()
        else:
            stream.close()
    elif ending == "error":
        with pytest.raises(ConnectionError) as raised:
            if asynchronous:
                await anext(stream)
            else:
                next(stream)
        assert raised.value is error
    elif asynchronous:
        async for _ in stream:
            pass
    else:
        list(stream)

    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1
    assert not [
        r
        for r in caplog.records
        if r.name == "opentelemetry.context" and r.levelno >= 40
    ]
